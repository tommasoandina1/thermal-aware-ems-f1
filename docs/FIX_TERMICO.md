# Fix del derating termico — cosa è cambiato e cosa devi aggiornare

## Il risultato che conta

Prima della modifica, nel forward pass **il raffreddamento non aveva alcun effetto**. Simulazione a 5 giri, rule-based, tre livelli di degrado del raffreddamento:

| cooling_factor | fuel [g] | shortfall [MJ] | SoC finale | T finale |
|---|---|---|---|---|
| 1.0 | 3871.1 | 5.863 | 0.461 | 50.9 |
| 0.5 | 3871.1 | 5.863 | 0.461 | 52.7 |
| 0.3 | 3871.1 | 5.863 | 0.461 | 55.2 |

Numeri identici a tre cifre decimali. La temperatura saliva, il derating veniva calcolato, e non toccava nulla di ciò che il powertrain erogava. Il progetto si chiama *Thermal-Aware* e il plant non lo era.

Dopo la modifica:

| cooling_factor | fuel [g] | shortfall [MJ] | SoC finale | T finale |
|---|---|---|---|---|
| 1.0 | 3865.1 | 5.990 | 0.461 | 50.9 |
| 0.5 | 3837.1 | 6.626 | 0.468 | 52.7 |
| 0.3 | 3554.6 | 13.660 | 0.641 | 55.0 |

Il vincolo termico ora entra nel bilancio di potenza: a raffreddamento degradato il tetto MGU-K si abbassa, l'elettrico non riesce a coprire la richiesta, lo shortfall raddoppia e il pacco resta carico perché non riesce a scaricarsi. È il comportamento fisicamente atteso, e prima non c'era.

---

## Le tre modifiche

### 1. `plant/battery.py` — il derating esce da `battery_step`

Prima:

```python
derating = thermal_derating_factor(Tbat_k, params)
P2_max = Uoc_k**2 /(4*R_int)*derating
if P2_k > P2_max:
    P2_k = P2_max
```

`Uoc²/(4·R_int)` non è un limite fisico: è il punto di massimo trasferimento di potenza del circuito equivalente, cioè metà OCV caduta su `R_int` — a 300 V sono 150 V di caduta e circa 15 kA. Vale 1.8–3.0 MW nella finestra SoC, quindi il valore derato scendeva sotto i 350 kW dell'MGU-K solo sopra ~58 °C, un grado sotto il limite di sicurezza.

Adesso resta come **guardia di solvibilità**, che è ciò che realmente è: oltre quel valore il discriminante della quadratica `U2² − Uoc·U2 + P2·R_int = 0` diventa negativo e `U2` va a NaN. Se scatta, emette un `RuntimeWarning` invece di clippare in silenzio — perché se scatta significa che il chiamante non ha applicato i limiti di componente a monte, e un clip silenzioso rompe il bilancio di potenza (richiesto ed erogato smettono di coincidere e lo shortfall di `powertrain()` smette di essere la fonte unica di verità).

Aggiunto anche `disc = np.maximum(0.0, ...)` prima della radice: esattamente al limite di solvibilità l'arrotondamento può spingere il discriminante di pochi ulp sotto zero, e quel NaN si propaga silenziosamente nell'integrazione del SoC. Stessa classe di problema dell'`np.inf` che propagava NaN attraverso `RegularGridInterpolator`.

`Tbat_k` resta nella firma: le chiamate posizionali esistenti continuano a funzionare, e il parametro servirà quando renderai `R_int` dipendente dalla temperatura.

### 2. `plant/powertrain.py` — il derating entra in `MGU_K`

```python
def MGU_K(P_mech_desired, E_deploy_acc_k, E_recharge_acc_k, params, dt, Tbat_k=None):
    derate = 1.0 if Tbat_k is None else thermal_derating_factor(Tbat_k, params)
    P_MGU_max = params['P_MGU_max'] * derate
```

e `powertrain(..., control_mode='u_split', Tbat_k=None)` lo inoltra.

Tre scelte da giustificare:

- **`Tbat_k=None` di default.** Retrocompatibile: i notebook 01–03 che non passano la temperatura mantengono esattamente i loro numeri pubblicati. Nessun risultato non termico da rifare. C'è un test che lo verifica.
- **Solo il deploy è derato, non la rigenerazione.** Il derating modella il limite termico lato scarica dell'MGU-K, come dichiara il tuo README. Impedire il recupero a caldo sarebbe un'affermazione di modellazione che niente nel progetto sostiene.
- **Il limite sta qui e non nella batteria.** Perché è qui che vive anche il limite regolamentare di 350 kW, ed è qui che il tetto derato partecipa al bilancio di potenza. Qualunque potenza il derating impedisce di erogare compare come shortfall, invece di sparire.

### 3. `rl/safety.py` e `rl/env.py` — allineamento

Il vincolo 2 del safety layer perde il fattore di derating (ora è solo la guardia di solvibilità); il vincolo 1 era già `P_MGU_max · derate` e ora coincide con il plant. `EMSEnv` passa `Tbat_k` a `powertrain()`.

**Le metriche RL non cambiano di una cifra**: il safety layer applicava già la semantica corretta. La verifica è utile in sé — conferma che i numeri RL erano già giusti e che ora sono giusti *e* coerenti col plant.

---

## Cosa devi aggiornare nei notebook

**Notebook 04.** Il backward pass usa già `P_MGU_K_max * derate_k`, quindi il bound del DP è corretto e resta invariato. I 261 binding events e lo stress test parametrico non cambiano: confrontano la policy col bound derato, meccanismo che non hai toccato.

Cambia la **validazione forward**. Ovunque chiami `powertrain(...)` nel loop di simulazione a valle della policy, aggiungi `Tbat_k=Tbat_profile[k]`:

```python
PI, P2, Pm, sf, Ed, Er, md = powertrain(
    P_gb[k], u_k, Ed, Er, params, dt,
    control_mode='P2', Tbat_k=Tbat_profile[k])
```

Dopo questa modifica il forward pass impone il vincolo che il backward pass assumeva. È esattamente il punto in cui prima c'era un model-plant mismatch invisibile: il DP si vincolava, la simulazione no.

Da rifare quindi: la tabella *before / after* (binding, shortfall, fuel) e i risultati dell'iterazione a punto fisso. Aspettati che i numeri *after* peggiorino leggermente a raffreddamento nominale e sensibilmente sotto stress — non è un peggioramento del metodo, è la scomparsa di uno sconto che non ti spettava. La riduzione dello shortfall del 99.1% andrà ricalcolata.

**Notebook 01–03.** Nulla. Non passano `Tbat_k`, quindi `derate = 1.0` e i risultati sono identici. Se vuoi, verificalo rieseguendo uno solo dei due DP single-lap e confrontando con 805.48 g.

**`Compare_Controllers.ipynb`.** Decidi se il confronto va fatto con o senza vincolo termico e dichiaralo. Il confronto pulito ECMS-vs-DP a SoC finale uguale (+1.8% carburante) è isotermo se nessuno dei due passa `Tbat_k`.

---

## Test

`tests/test_plant_thermal.py`, 11 test, tutti passano. Insieme ai 23 del layer RL fanno 34.

| test | proprietà bloccata |
|---|---|
| `test_derating_caps_the_mguk_deploy_envelope` | a 50 °C il tetto è 2/3 di 350 kW, a 55 °C 1/3, a 60 °C zero |
| `test_derating_is_monotone_in_temperature` | monotonia |
| `test_regeneration_is_not_derated` | il recupero a caldo resta possibile |
| `test_no_temperature_means_no_thermal_limiting` | retrocompatibilità dei notebook 01–03 |
| `test_battery_step_no_longer_thermally_clips` | il derating è uscito davvero dalla batteria |
| `test_solvability_guard_warns_instead_of_clipping_silently` | il clip è rumoroso, non silenzioso |
| `test_no_warning_in_the_normal_operating_range` | la guardia non spara mai in esercizio normale |
| `test_thermal_limit_shows_up_as_shortfall` | **la ragione della modifica**: il vincolo raggiunge il bilancio di potenza |
| `test_power_balance_closes_when_hot` | `shortfall == P_gb − P_ICE − P_mech` esattamente, a ogni temperatura |
| `test_deploy_budget_and_derating_compose` | due limiti attivi danno il più stretto, non il prodotto |
| `test_thermal_runaway_is_self_limiting` | a piena richiesta con raffreddamento al 30% la temperatura converge al limite invece di divergere |

L'ultimo è quello che vale di più: è un test closed-loop su 4000 step che fallirebbe se il derating tornasse a non raggiungere il plant.

---

## Come applicare

```bash
# opzione A: patch
git apply thermal_derating.patch

# opzione B: sostituzione diretta
cp thermal_fix/plant/battery.py plant/
cp thermal_fix/plant/powertrain.py plant/
cp thermal_fix/rl/safety.py thermal_fix/rl/env.py rl/
cp thermal_fix/tests/test_plant_thermal.py tests/

python -m pytest tests/ -q      # 34 test
```

Poi aggiungi `Tbat_k=` alle chiamate a `powertrain()` nel forward pass del notebook `04` e rigenera i risultati termici.

---

## Una riga per il README

Nella sezione delle semplificazioni, sostituisci la voce sul derating con qualcosa del genere:

> Il derating termico è applicato all'inviluppo di deploy dell'MGU-K (`P_MGU_max · derate(T_bat)`), dove risiede anche il limite regolamentare di 350 kW, così che la potenza impedita dal vincolo termico compaia esplicitamente come shortfall nel bilancio di potenza invece di essere assorbita silenziosamente dal modello di batteria. Il derating non è applicato alla rigenerazione. Il tetto `Uoc²/(4·R_int)` del circuito equivalente è mantenuto come guardia di solvibilità, non come limite fisico.
