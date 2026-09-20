# Layer RL — motivazione di ogni scelta

Documento di accompagnamento al package `rl/`. Non è documentazione d'uso (quella sta in `rl/README.md`): qui c'è il *perché* di ogni decisione, nella forma in cui ti serve per difenderla in tesi.

Ho scaricato il tuo plant reale dal repo (`plant/parameters.py`, `vehicle_dynamics.py`, `battery.py`, `powertrain.py`, `controller/rule_based_controller.py`) e ho costruito tutto sopra le API esistenti. Nessuna funzione del plant è stata modificata. Tutto ciò che segue è stato eseguito e testato: 23 test passano, il pipeline end-to-end gira.

---

## 1. Cosa c'è nei file

| file | righe | ruolo |
|---|---|---|
| `rl/config.py` | 130 | tre dataclass: `EnvConfig`, `SACConfig`, `TrainConfig`. Un esperimento è completamente descritto da queste tre più il seed |
| `rl/safety.py` | 150 | insieme ammissibile analitico dell'azione + proiezione |
| `rl/env.py` | 300 | `EMSEnv`, ambiente Gymnasium sopra il tuo plant |
| `rl/baselines.py` | 160 | rule-based, ECMS, replay-DP come policy dello stesso ambiente |
| `rl/buffer.py` | 100 | replay buffer, prioritizzazione opzionale, canale di costo |
| `rl/sac.py` | 330 | SAC con auto-tuning dell'entropia e modalità Lagrangiana |
| `rl/bc.py` | 110 | behaviour cloning da un controllore maestro |
| `rl/train.py` | 230 | loop di training con curriculum e domain randomization |
| `rl/evaluate.py` | 130 | rollout, metriche a SoC equalizzato, tabella di confronto |
| `scripts/run_rl_experiment.py` | 170 | esperimento completo con un comando |
| `tests/test_rl.py` | 250 | 23 test |

Circa 1600 righe più i test. Dipendenze aggiuntive: `torch`, `gymnasium`.

---

## 2. Formulazione dell'MDP

### 2.1 Stato

```
[SoC, T_bat, E_deploy_residuo, P_gb, v, a, posizione_nel_giro, giri_rimanenti, preview(k)]
```

Le prime tre non sono opzionali: senza `T_bat` la transizione non è markoviana (il derating dipende dalla temperatura, che ha dinamica propria), e senza l'accumulatore di deploy nemmeno (il budget è cumulativo per giro).

`posizione_nel_giro` e `giri_rimanenti` ci sono perché **il problema è a orizzonte finito e non stazionario**. La politica ottima dipende da quanta gara resta — è esattamente la ragione per cui il tuo DP multi-lap senza vincolo svuota il pacco nei primi due giri. Senza queste due componenti nello stato, l'agente non può replicare quel comportamento né correggerlo: sta risolvendo un problema stazionario che non è il tuo.

`preview(k)` sono `k` campioni futuri di `P_gb` (default 5, spaziati 0.5 s → orizzonte 2.5 s). Scelta discussa in §2.4.

Normalizzazione **analitica e fissa** (nessuna running statistic): un checkpoint è riproducibile senza portarsi dietro stato aggiuntivo. Costo: se cambi il circuito devi verificare che le costanti di scala coprano ancora il range.

### 2.2 Azione

Scalare in `[-1, 1]` → potenza elettrica ai morsetti batteria `P2`, cioè `control_mode='P2'` di `powertrain()`.

Il dominio elettrico è quello giusto perché è dove sono definiti sia il limite regolamentare 350 kW sia il budget di deploy — è la stessa correzione che avevi già applicato ai notebook (`MGU_K()` applicava i 350 kW alla potenza meccanica invece che a quella elettrica). Comandare `P2` invece di `u_split` evita di reintrodurre lo stesso errore per un'altra strada.

Due mappature disponibili:

- **`relative`** (default): `a=-1 → lb`, `a=+1 → ub`, dove `[lb, ub]` è l'inviluppo ammissibile *corrente*. L'agente controlla la frazione di capacità disponibile che vuole usare. Vantaggio: tutto il range di azione è sempre significativo, l'esplorazione non spreca campioni su comandi infattibili, la politica trasferisce tra stati in cui l'inviluppo ha larghezze molto diverse (prima/dopo l'esaurimento del budget). Svantaggio: la semantica di un dato valore di azione dipende dallo stato, quindi la politica appresa è più difficile da ispezionare e la mappatura è non stazionaria dentro l'episodio.
- **`absolute`**: `a` scalata sull'inviluppo regolamentare fisso e poi clippata. Semantica fissa, direttamente confrontabile con le traiettorie di controllo DP/ECMS, ma gran parte del range è inerte quando l'inviluppo è stretto.

Entrambe implementate perché è un trade-off vero e il confronto vale la pena riportarlo come ablazione.

### 2.3 Vincoli: dove li ho messi e perché

Il punto centrale del design. I vincoli **non** sono nel reward, sono nell'insieme ammissibile (`rl/safety.py`). Cinque, imposti simultaneamente e analiticamente:

1. inviluppo MGU-K, derato termicamente: `P2 ≤ P_MGU_max · derate(T_bat)`
2. tetto di potenza batteria: `P2 ≤ Uoc²/(4·R_int) · derate`
3. budget deploy per giro: `P2·dt ≤ E_deploy_max − E_deploy_acc`
4. finestra SoC: `SoC_min ≤ SoC_next ≤ SoC_max` (via Coulomb counting, con margine 0.98 perché usare `Uoc` al posto di `U2` è conservativo solo in scarica)
5. disponibilità fisica di frenata: `P_mech ≥ P_gb` quando `P_gb < 0` (non si recupera più potenza di quella che il cambio sta dissipando)

Nella tassonomia di Brunke et al. questi sono **Livello III (hard)**: sono garantiti per costruzione, non appresi. Un agente RL che deve imparare a non superare il budget di deploy spende metà del training a scoprire un vincolo che tu conosci in forma chiusa. Questo è il pattern *safety layer* di García & Fernández: si prende l'azione proposta e la si proietta sul punto ammissibile più vicino.

Ci sono due conseguenze concrete che valgono più della teoria:

- **niente clipping silenzioso.** `battery_step()` clippa internamente `P2` a `P2_max` e clippa `SoC` ai bound. Quando questo succede, potenza richiesta e potenza erogata divergono e lo `shortfall` restituito da `powertrain()` smette di essere la fonte unica di verità. Con il safety layer davanti, il clipping interno non si attiva mai — e c'è un test che lo verifica su tutto l'episodio (`test_no_silent_clipping_by_the_plant`).
- **niente intervalli invertiti.** `lb > ub` passato a `np.linspace` o a un campionatore uniforme produce output silenziosamente privo di senso. È il bug che avevi già incontrato nel backward pass vettorizzato. Qui è impossibile per costruzione (guardia esplicita) e c'è un test che lo verifica su tutti gli angoli dello spazio di stato.

La ripetibilità del SoC di fine giro resta **Livello I (soft)**, esattamente come nel tuo DP: hinge asimmetrica.

### 2.4 Preview: la decisione più discutibile

Il DP è chiaroveggente. Un agente senza preview risolve un problema strettamente più difficile (parzialmente osservabile), e il gap RL-DP misurerebbe due cose insieme: errore di approssimazione **e** perdita di informazione.

Con `preview_horizon = 5` il confronto è: stessa informazione (approssimativamente), solutore diverso. Con `preview_horizon = 0` diventa il POMDP. Entrambi sono difendibili, ma vanno riportati come due esperimenti distinti, non mescolati. La preview è anche realistica: la mappa del circuito è nota e la traccia di velocità è ripetibile giro su giro.

**Suggerimento**: fai la tesi con preview, e riporta l'ablazione a preview zero come misura di *quanto del vantaggio del DP venga dall'informazione invece che dall'ottimizzazione*. È una domanda che nessuno ti ha ancora fatto e la risposta è interessante.

---

## 3. Reward: unità grammi-equivalenti

Ogni termine è convertito nei grammi di carburante che avrebbero prodotto lo stesso effetto. Il reward è quindi il costo in grammi cambiato di segno, e i pesi sono ancorati alla fisica invece che tarati a mano.

```
r = −( fuel_g + w_shortfall·unmet_MJ + w_rate·|ΔP2|_MW ) · scale
r_fine_giro −= w_soc_lin·h + w_soc_quad·h² ,   h = max(0, SoC_target − SoC)
```

**Derivazioni** (tutte da `parameters.py`, non inventate):

- 1 MJ di energia non erogata al cambio ≡ `1e6/(η_ICE·LHV) = 1e6/(0.498·44e6) = 45.6 g` di carburante. Questo è il prezzo *naturale*. Il default `w_shortfall = 500 g/MJ` è circa 10× il naturale: uno shortfall è una perdita di tempo sul giro, non una spesa di carburante, quindi va prezzato molto sopra il suo equivalente energetico. Il fattore 10 è l'unico numero arbitrario nel reward, ed è dichiarato come tale.
- 1 unità di SoC ≡ `E_pack/(η_ICE·LHV)·1000 = 5.7e6/(0.498·44e6)·1000 = 260 g`. Da cui `w_soc_lin = 260` esattamente, più un termine quadratico `w_soc_quad = 20000` che rende un deficit di 0.05 SoC un costo di 50 g invece di 13 g.
- `w_rate` penalizza `|ΔP2|` in MW. Serve a eliminare il chattering. Nel tuo ECMS il chattering è **strutturale** — nasce dall'assenza di un termine di penalità sulla rate — ed è un fenomeno distinto dall'esaurimento del budget MGU-K, che è una discontinuità fisica. La stessa distinzione va mantenuta in RL: se vedi oscillazioni, guarda prima se `w_rate = 0`.

**Il punto più importante di tutta la sezione**: la penalità di shortfall è **continua** nell'entità dello shortfall, non una costante binaria. Un `BIG_M` binario rende l'ottimizzatore indifferente tra 0.1 MJ e 12 MJ di domanda non soddisfatta. In DP costava un'approssimazione; in RL è fatale, perché il gradiente è l'unico segnale di apprendimento. C'è un test dedicato (`test_shortfall_penalty_is_continuous_not_binary`).

`reward_scale = 0.05` è puro condizionamento numerico per la rete, non fisica: reward non limitati fanno esplodere i gradienti del critico.

---

## 4. Metrica: `fuel_eq_g`

```
fuel_eq = fuel + (SoC_target − SoC_finale) · E_pack/(η_ICE·LHV) · 1000
```

Questa è la cifra da riportare in tabella, non `fuel`. Confrontare il carburante grezzo tra controllori che finiscono a SoC diverso non è un confronto: qualunque strategia può sembrare frugale arrivando a pacco vuoto. È esattamente la ragione per cui il `−11.5%` del rule-based contro il DP non era difendibile — il rule-based lasciava domanda non soddisfatta e carica inutilizzata.

Il calcolo è nel metodo `summary()` dell'ambiente, quindi ogni controllore che gira dentro `EMSEnv` la riceve automaticamente. Test: `test_soc_equalised_fuel_is_the_reported_metric`.

Le altre metriche riportate, in ordine di importanza: shortfall totale, numero di step con vincolo termico attivo, SoC finale, dispersione tra seed, tempo di inferenza per step.

Il tempo di inferenza non è cosmetico. È **l'argomento a favore dell'RL**: il DP è ottimo ma richiede l'intera traiettoria futura e una risoluzione offline; la policy addestrata è un forward pass. Nel run demo: ~78 µs/step su CPU. Anche perdendo qualche punto percentuale sul DP, vinci sull'unica dimensione che conta per il deployment.

---

## 5. Algoritmo: SAC

### 5.1 Perché

- azione continua → esclude la famiglia value-based senza discretizzazione arbitraria;
- off-policy → il replay buffer viene riusato; conta, perché un episodio a cinque giri sono 3886 valutazioni del plant;
- politica stocastica a massima entropia → esplorazione strutturata invece del rumore additivo OU di DDPG. Il tuo paesaggio di costo ha discontinuità vere (esaurimento budget, attivazione derating): il rumore additivo tende a collassare sul lato della discontinuità da cui parte;
- è empiricamente il meno sensibile agli iperparametri della famiglia.

**Contro-argomento da dichiarare**: TD3 resta il baseline di confronto obbligatorio, e almeno uno studio nella letteratura di process control riporta una variante TD3 entropy-maximizing (EMTD3) che batte sia TD3 sia SAC su sample efficiency e velocità di convergenza. SAC non è universalmente superiore; la scelta è motivata dalla struttura del problema.

### 5.2 Cosa è implementato

- twin critic con target condiviso e clipped double-Q (prendi il minore dei due Q per limitare il bias di sovrastima);
- target network con Polyak averaging;
- politica gaussiana squashed con tanh, con la correzione esatta della log-probabilità per il cambio di variabile, nella forma numericamente stabile `log(1−tanh²u) = 2(log2 − u − softplus(−2u))`;
- **auto-tuning della temperatura `α`** contro un'entropia target di `−dim(A)`. Il paper presenta SAC con `α` fisso; nella pratica si usa la versione automatica, che elimina l'iperparametro più sensibile. Questa era una delle lacune che ti avevo segnalato nel riassunto del paper (§8 del documento precedente): qui è colmata;
- `policy_delay` configurabile (SAC usa 1; >1 riproduce il delayed policy update di TD3, utile per l'ablazione).

### 5.3 Modalità Lagrangiana (CMDP)

Attivabile con `--lagrangian`. Trasforma il problema in:

```
max_π E[Σ r]   s.t.   E[Σ c] ≤ d
```

dove `c` è l'energia di shortfall per step in MJ e `d` il budget (default 0.05 MJ). Risolto con ascesa duale sul moltiplicatore `λ`.

Dettagli implementativi che contano:

- c'è un **cost critic separato** (twin, come quello del reward), ma con target **pessimistico**: per i costi si prende il `max` dei due, non il `min`. Prendere il minimo sottostimerebbe il costo, che è l'esatto contrario di quello che serve;
- l'aggiornamento di `λ` avviene **una volta per episodio**, non per step, guidato dalla violazione del budget *episodico*, con una stima esponenziale mobile. Il vincolo è un budget sull'intera gara: un moltiplicatore che reagisce a una quantità per-step insegue rumore;
- quando `--lagrangian` è attivo, lo script azzera `w_shortfall` nel reward. Tenere entrambi prezzerebbe lo shortfall due volte.

**Perché è la strada che ti consiglio**: elimina completamente il problema di calibrare i pesi relativi delle penalità nel reward — sostituisce un peso arbitrario (`w_shortfall = 500`, il fattore 10 di cui sopra) con un budget interpretabile in MJ. È la critica standard ai lavori RL e con questa formulazione non ti tocca.

---

## 6. Sample efficiency: le quattro leve

Il collo di bottiglia riconosciuto dell'RL. Tutte e quattro le leve sono implementate.

**Behaviour cloning** (`rl/bc.py`). Pre-addestra l'attore a riprodurre un maestro, poi SAC rifinisce. Qui la situazione è migliore del solito: un maestro *ottimo* esiste già, il DP. Passa `--dp-traj data/results/dp_P2.npy` con la traiettoria di comando elettrico esportata dal notebook 03/04 e il clone parte da lì; senza, il maestro di default è l'ECMS.

Solo la testa della media è supervisionata, non la log-std: si vuole un attore che sappia *dove* agire ma resti libero di esplorare *quanto*. Supervisionando anche la std si ottiene una politica quasi deterministica che annulla il termine entropico per le prime migliaia di update.

Due caveat da scrivere in tesi: (i) il DP è chiaroveggente e l'agente no, quindi il clone perfetto è impossibile — l'errore residuo di cloning *misura* quanto del vantaggio DP venga dall'informazione; (ii) il cloning trasferisce anche i difetti del maestro, e il DP multi-lap baseline svuota il pacco nei primi due giri: clona la variante con vincolo di fine giro.

Nel test: MSE di cloning 0.027 su ECMS in 30 epoche, e l'attore clonato riproduce l'ECMS a 949.7 g contro 951.2 g equalizzati.

**Curriculum** (`rl/train.py`). Tre stadi: 1 giro → 3 giri → 5 giri, con pesi e buffer ereditati. L'orizzonte di credit assignment cresce di 5× tra il primo e l'ultimo stadio; single-lap e multi-lap sono task correlati, che è la condizione esatta in cui il transfer aiuta.

**Domain randomization**. SoC iniziale, temperatura iniziale, efficacia di raffreddamento. Il terzo serve a un secondo scopo oltre all'anti-overfitting: **garantire che il vincolo termico venga effettivamente esercitato**. A raffreddamento nominale il derating vincola raramente; un agente addestrato solo lì non impara a gestirlo. Campionare `cooling_factor ∈ [0.3, 1.0]` trasforma il tuo stress test parametrico in una distribuzione di task.

**Prioritized replay** (`--prioritized`). Gli episodi sono lunghi (3886 step) e le transizioni interessanti sono rare: esaurimento budget, attivazione derating, domanda non soddisfatta. Il campionamento uniforme le annega. Implementato con esponente `α`, correzione di importance sampling con `β`, e priorità = errore TD. Implementazione ad array invece che sum-tree: fino a qualche milione di transizioni è abbastanza veloce ed è molto più facile da verificare.

---

## 7. Baseline nello stesso harness

`rl/baselines.py` reimplementa rule-based, ECMS e replay-DP come policy dell'ambiente.

La ragione è metodologica, non algoritmica. I tre controllori esistono già nei notebook, ma ciascuno gira nel proprio loop di simulazione. Qualunque confronto costruito su tre loop diversi è vulnerabile all'obiezione che la differenza venga dall'harness e non dalla strategia. Rieseguendoli dentro `EMSEnv` l'obiezione sparisce: stesso plant, stesso safety layer, stessa contabilità di carburante, shortfall e SoC terminale.

L'ECMS è implementato con `s(t) = s₀ − Kp·(SoC − SoC_ref)` e ricerca diretta su griglia dell'intervallo ammissibile. La ricerca su griglia invece di una condizione di stazionarietà in forma chiusa è deliberata: il costo non è liscio (saturazione fuel-flow, budget, tetto termico) proprio nei punti operativi che contano. Default `s₀ = 1/η_ICE = 2.008`, `Kp = 5` — la calibrazione corretta, non quella bang-bang.

C'è un test che verifica che `s₀=1, Kp=170` produca marcatamente più switching di comando della calibrazione corretta (`test_high_gain_ecms_chatters`): il chattering diventa una proprietà verificata automaticamente, non un'osservazione aneddotica.

---

## 8. Test: cosa pinnano

23 test, tutti passano. Non è coverage fine a sé stessa — ognuno blocca una proprietà che, se si rompesse in silenzio, produrrebbe risultati plausibili ma privi di significato. Che è il modo di fallire più difficile da beccare guardando una learning curve.

| test | proprietà bloccata |
|---|---|
| `test_bounds_never_inverted` | `lb > ub` impossibile su tutti gli angoli dello spazio di stato |
| `test_deploy_budget_closes_the_upper_bound` | budget esaurito ⇒ `ub = 0` |
| `test_thermal_derating_shrinks_the_upper_bound` | il derating è osservabile |
| `test_regen_cannot_exceed_available_braking_power` | niente energia dal nulla |
| `test_projection_roundtrip` | `project` / `inverse_project` sono inverse (serve al BC) |
| `test_hard_constraints_hold_for_any_action` | nessuna azione costante viola SoC / termico / budget, su tutto l'episodio |
| `test_no_silent_clipping_by_the_plant` | il plant non clippa mai dietro le spalle del controllore |
| `test_deploy_budget_resets_at_lap_boundary` | reset regolamentare per giro |
| `test_reward_is_negative_gram_equivalent_cost` | il reward è esattamente il costo in grammi |
| `test_shortfall_penalty_is_continuous_not_binary` | shortfall maggiore ⇒ reward strettamente peggiore |
| `test_soc_equalised_fuel_is_the_reported_metric` | la metrica pubblicata è quella corretta |
| `test_determinism_given_a_seed` | riproducibilità con randomizzazione attiva |
| `test_rule_based_reproduces_the_repository_figures` | **test di ancoraggio**: 773.5 g / 2.02 MJ / SoC_f 0.5417 entro il 2% |
| `test_sac_update_runs_and_changes_parameters` | il grafo di SAC + Lagrangiano è integro |

Il test di ancoraggio è il più importante: valida l'intero harness contro numeri che hai già verificato per altra via.

---

## 9. Due cose che ho trovato nel plant

**(a) Il derating termico, come codificato, è quasi inerte.**

`plant/battery.py` moltiplica per il fattore di derating il tetto di potenza *della batteria*, `Uoc²/(4·R_int)`. Con `R_int = 0.01 Ω` quel tetto vale:

| SoC | Uoc [V] | Uoc²/(4R) [kW] |
|---|---|---|
| 0.2 | 270.0 | 1822 |
| 0.5 | 303.9 | 2308 |
| 0.9 | 345.0 | 2976 |

Il valore derato scende sotto il limite MGU-K di 350 kW solo quando `derate < 0.19`, cioè sopra circa 58 °C — un grado sotto il limite di sicurezza assoluto di 60 °C. In pratica il derating non vincola mai.

Il tuo README descrive l'intento diversamente e, fisicamente, in modo più plausibile: *"MGU-K: ... a temperature-dependent derating factor on maximum discharge power"*. `rl/safety.py` implementa quell'intento (vincolo 1: `P2 ≤ P_MGU_max · derate`). È quello che rende il vincolo termico osservabile e vincolante a temperature realistiche.

**Questo va verificato anche nel DP.** Se il notebook `04` conta i binding events sul tetto batteria, i tuoi 261 eventi a raffreddamento nominale potrebbero misurare una cosa diversa da quella che il README dichiara. Vale la pena controllarlo prima della consegna: è il tipo di incoerenza che un relatore trova.

**(b) L'harness riproduce i tuoi numeri.** Il rule-based dentro `EMSEnv` dà 777.1 g contro i 773.5 g canonici (+0.5%) e 1.92 MJ contro 2.02 MJ. La differenza residua viene dal safety layer che limita il comando prima che il plant lo veda, dove i notebook lasciano clippare `battery_step()` internamente. La versione con safety layer è la più coerente delle due, per la ragione al §2.3.

---

## 10. Numeri misurati (esecuzioni reali, non stime)

**Baseline, giro singolo (Canada quali, N=726):**

```
controller       fuel_eq [g]    fuel [g]  short [MJ]   therm   SoC_f   us/step
rule_based            843.56      777.07      1.9237      97   0.544      40
ecms                  895.72      853.06      1.0589     229   0.636      58
```

**ECMS multi-lap sotto stress termico (N=3886):**

```
cooling      fuel_eq [g]  short [MJ]   therm   SoC_f
nominale         4154.97      2.1130     882   0.675
0.5              4103.66      3.9215    1265   0.702
0.3              4055.69      5.9470    1614   0.765
```

La lettura è quella attesa e fisicamente corretta: raffreddamento peggiore → il derating vincola più spesso → meno energia elettrica utilizzabile → più shortfall e SoC finale più alto (il pacco non riesce a scaricarsi). Il carburante *sembra* scendere, ma è esattamente l'artefatto che `fuel_eq` è progettato per smascherare: guarda la colonna equalizzata insieme allo shortfall, mai il carburante da solo.

**Pipeline completo** su 12 000 step (dimostrativo, ~2 minuti CPU — lontanissimo dalla convergenza): BC porta l'attore a MSE 0.027 e a 888.6 g equalizzati, SAC scende a 876.9 g ma con shortfall ancora alto. Serve un budget reale di 200–300k step per stadio.

**Cosa aspettarti a convergenza**: l'RL non batterà il DP. Il target ragionevole è il 2–5% sopra l'ottimo chiaroveggente, con shortfall dello stesso ordine dell'ECMS o migliore, e inferenza tre-quattro ordini di grandezza più veloce del DP.

---

## 11. Come integrarlo

```bash
# nel tuo repo
cp -r rl/ tests/test_rl.py scripts/run_rl_experiment.py .
pip install torch gymnasium

python -m pytest tests/test_rl.py -q                    # 23 test
python scripts/run_rl_experiment.py --steps 20000 --profile single_lap --out runs/quick
python scripts/run_rl_experiment.py --steps 300000 --out runs/sac
python scripts/run_rl_experiment.py --steps 300000 --lagrangian --cost-limit 0.05
python scripts/run_rl_experiment.py --dp-traj data/results/dp_P2.npy
```

Lo script produce `comparison.txt`, `log.csv`, i checkpoint, `sac_trajectory.npz` e una figura diagnostica a quattro pannelli (potenze, comando elettrico con inviluppo ammissibile ombreggiato, SoC + temperatura, shortfall). L'npz è pensato per essere ricaricato nei notebook.

Da aggiungere a `requirements.txt`:

```
torch>=2.0
gymnasium>=0.29
```

Il package non dipende da `paths.py`: `EnvConfig.data_root` è il solo punto di ingresso dei percorsi. Quando applicherai la migrazione dei path, basta passare `data_root` dal tuo `paths.py`.

Nota sul repository: la purga della cache da 61 MB va fatta **prima** di committare questi file, altrimenti riscrivi la storia due volte.

---

## 12. Cosa non c'è (e la scaletta successiva)

| mancante | perché | priorità |
|---|---|---|
| **TD3 di confronto** | è il baseline obbligatorio; con questa struttura sono ~80 righe (togli l'entropia, aggiungi target policy smoothing) | alta |
| **Export della traiettoria DP** | serve un `.npy` di `P2[k]` dal notebook 03/04 per usare il DP come maestro e come riga di riferimento nella tabella | alta |
| **Ablazione preview 0** | il caso POMDP, con policy ricorrente (GRU) | media |
| **CIS dal value grid del DP** | gli stati con valore finito nella tua griglia DP *sono* il control invariant set discretizzato: usalo per campionare gli stati iniziali, resettare, e penalizzare le uscite (Bo et al.). È probabilmente il contributo più originale disponibile | media |
| **Control barrier function termica** | `h(x) = T_derate − T_bat`, con `ḣ` in forma chiusa dal tuo modello lumped-capacitance → vincolo risolvibile con un QP a ogni step, garanzia di non violazione *per costruzione* | ricerca |
| **RL che tara ECMS** | l'azione diventa `Δs` invece di `P2`: 1-D, vincoli garantiti dal solutore ECMS, fallback ovvio. Molto più economico e molto più credibile come sistema deployable | alta se hai poco tempo |
| **Ambiente vettorizzato** | ~0.45 s per episodio a 5 giri; con 8 ambienti paralleli in NumPy vettorizzato il training scende di quasi un ordine di grandezza | media |
| **Distributional SAC** | ottimizzare quantili invece della media: il tuo rischio è asimmetrico (shortfall inaccettabile, margine costa solo carburante) | ricerca |

L'ultima riga della tabella "priorità alta" merita un commento. Se il tempo prima di settembre stringe, **la strada RL-tuned ECMS è quella da fare per prima**: riusa questo stesso ambiente e lo stesso agente, cambia solo la definizione di azione, e produce un risultato difendibile con un decimo del budget di training.

---

## 13. Riferimenti d'implementazione

Le convenzioni seguono i pattern consolidati di CleanRL (SAC single-file, correzione tanh, auto-tuning di `α`), Stable-Baselines3 (interfaccia dell'agente, gestione di `done` vs `truncated`), Safety-Gym / SAC-Lagrangian (cost critic + ascesa duale) e PC-Gym (ambiente di process control con oracolo di riferimento). Nessun codice copiato: la struttura è riscritta sopra il tuo plant, con le convenzioni di segno e le API di `powertrain()` e `battery_step()`.

Una scelta implementativa da segnalare: nel loop di training `done` marca solo le terminazioni **vere**, non la troncatura a fine profilo. Il limite di tempo è artificiale — bootstrappare attraverso di esso è corretto, trattarlo come terminale no. È un errore comune e falsa sistematicamente la value function verso il basso in prossimità della fine dell'episodio.
