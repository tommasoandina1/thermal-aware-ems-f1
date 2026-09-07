# Guida allo studio — RL e il codice del layer `rl/`

Documento di comprensione. Non è documentazione d'uso (`rl/README.md`) né giustificazione delle scelte (`SCELTE_RL.md`): serve a farti capire, in ordine, la teoria che ti serve, il codice che la implementa, e a verificare da solo di aver capito.

---

## Come usare questo documento

Tre passaggi, in quest'ordine. Saltare il primo è la ragione più comune per cui il codice RL sembra opaco.

1. **Eseguire e strumentare** (§4). Guardare i numeri prima delle astrazioni.
2. **Leggere con una domanda in mano** (§2 e §3). Ogni sezione ha in fondo le domande a cui deve permetterti di rispondere.
3. **Rompere di proposito** (§4.2). I test sono la specifica; romperli ti dice cosa proteggevano.

Le domande sono raccolte tutte insieme in §7, con le risposte in §8. Usa §8 solo dopo aver provato.

Tempo realistico: mezza giornata per §1–§4, un'altra mezza per §5–§6.

---

# 1. Teoria RL, ancorata al tuo problema

Tutta la teoria qui sotto è istanziata sul tuo powertrain. Se hai già letto il riassunto del paper di Devarakonda, questa è la stessa materia vista dal lato dell'implementazione.

## 1.1 L'MDP è già il tuo problema

Un MDP è la quadrupla `(S, A, P, R)`. Nel tuo caso:

| simbolo | cos'è nel tuo progetto |
|---|---|
| `S` | `(SoC, T_bat, E_deploy residuo, punto operativo, fase di gara)` |
| `A` | potenza elettrica `P2` ai morsetti batteria, continua |
| `P(s,a,s')` | il tuo plant quasistatico: `powertrain` + `battery_step` + `thermal_model` |
| `R(s,a)` | meno il carburante bruciato, meno le penalità |

La **proprietà di Markov** richiede che il futuro dipenda solo dallo stato corrente:

```
P(s_{t+1} | s_t, a_t, s_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

Questo è un vincolo *di progetto*, non una proprietà che scopri. Se ometti `T_bat` dallo stato, il plant continua a funzionare ma l'agente non può prevedere il derating: sta risolvendo un problema stocastico dove il tuo è deterministico, e la varianza apparente che vede è in realtà informazione che gli hai nascosto. Stesso discorso per `E_deploy` residuo.

## 1.2 Il tuo DP è già l'equazione di Bellman

```
V*(s)    = max_a  E[ R(s,a) + γ V*(s') ]
Q*(s,a)  = E[ R(s,a) + γ max_{a'} Q*(s', a') ]
```

Il tuo `backward_process` risolve la prima **esattamente**, su una griglia discretizzata `(SoC, E_deploy)`, con modello noto e conoscenza perfetta del futuro. L'RL risolve la stessa equazione **per approssimazione**, con reti neurali al posto della griglia e campionamento al posto del modello.

Da qui discende il framing corretto di tutta la tua tesi:

> Il DP non è un concorrente dell'RL, è il suo **oracolo**. Il gap RL–DP misura l'errore di approssimazione. L'RL non può vincere, perché il DP ha informazione che l'RL non ha.

E il corollario che rende il capitolo interessante: se dai all'agente una finestra di preview, riduci il divario informativo, e il gap residuo diventa una misura pulita del solo errore di approssimazione. L'ablazione preview-sì / preview-no separa le due cause. È una domanda a cui il tuo DP da solo non può rispondere.

## 1.3 Funzioni valore, e perché ne servono due

- `V^π(s)` = ritorno atteso partendo da `s` e seguendo `π`
- `Q^π(s,a)` = ritorno atteso partendo da `s`, facendo `a`, poi seguendo `π`
- `A^π(s,a) = Q^π(s,a) − V^π(s)` = quanto quell'azione è meglio della media

Perché `Q` e non solo `V`: per scegliere l'azione da `V` ti servirebbe il modello (per sapere dove ti porta `a`). `Q` incorpora già quel passo, quindi rende possibile il controllo **model-free**. È tutta la differenza tra il tuo DP (che usa il plant nel backward pass) e l'RL (che non lo usa mai, se non come simulatore da cui campionare).

## 1.4 On-policy vs off-policy, e perché ti importa

- **On-policy** (PPO, TRPO): impari sulla politica che stai usando. Dopo ogni aggiornamento i dati raccolti sono obsoleti e vanno buttati.
- **Off-policy** (DQN, DDPG, TD3, SAC): impari su dati raccolti da qualunque politica, tramite un replay buffer.

Nel tuo caso un episodio a cinque giri sono 3886 valutazioni del plant. Buttarle dopo un update è insostenibile. Questo, da solo, esclude PPO.

Off-policy è anche ciò che rende possibili due cose che usi: il **warm-start per behaviour cloning** (i dati del maestro finiscono nello stesso buffer) e l'**offline pre-training** dalle traiettorie DP.

## 1.5 SAC, derivato invece che citato

**Il problema che risolve.** Un attore deterministico (DDPG) esplora aggiungendo rumore. Se il paesaggio di costo ha una discontinuità — nel tuo caso l'esaurimento del budget di deploy, o l'attivazione del derating — il rumore additivo tende a restare sul lato da cui è partito.

**L'idea.** Cambia l'obiettivo: non massimizzare solo il ritorno, ma il ritorno **più l'entropia** della politica.

```
J(π) = E[ Σ_t γ^t ( R(s_t,a_t) + α · H(π(·|s_t)) ) ]
```

dove `H(π(·|s)) = E_{a~π}[ −log π(a|s) ]`. Il parametro `α` (temperatura) pesa "quanto vuoi restare incerto". L'esplorazione diventa parte dell'obiettivo invece che un'aggiunta euristica.

**Conseguenza sulle equazioni.** Il target di Bellman acquista il termine entropico ("soft Bellman"):

```
y = r + γ ( min_{j=1,2} Q_targ,j(s', a') − α log π(a'|s') ),   a' ~ π(·|s')
```

e l'attore massimizza

```
E_{a~π}[ min_j Q_j(s,a) − α log π(a|s) ]
```

**Il `min` sui due critici** (clipped double-Q, ereditato da TD3): due reti `Q` indipendenti, si usa la più pessimista. Serve a limitare il bias di sovrastima — l'errore di approssimazione è rumore a media circa nulla, ma il `max` dell'operatore di Bellman lo trasforma sistematicamente in sovrastima, che si accumula.

**Il tanh e la sua correzione.** L'azione deve stare in `[-1,1]`, quindi si campiona `u` da una gaussiana e si applica `a = tanh(u)`. Ma allora la densità di `a` non è quella di `u`: serve il jacobiano del cambio di variabile.

```
log π(a|s) = log N(u; μ, σ) − Σ_i log(1 − tanh²(u_i))
```

Nel codice il secondo termine è nella forma numericamente stabile `2(log2 − u − softplus(−2u))`, perché `log(1 − tanh²u)` va a `-inf` per `|u|` grande.

**L'auto-tuning di `α`.** Invece di fissarlo, lo si risolve come vincolo: mantieni l'entropia media sopra un target `H̄`. La discesa duale dà

```
L(α) = −α · ( log π(a|s) + H̄ )
```

Con `H̄ = −dim(A)`, che per te è `−1`. Se la politica è troppo deterministica `α` cresce, e viceversa. Elimina l'iperparametro più sensibile. **Non è nel paper di Devarakonda**, che presenta SAC con `α` fisso: viene da Haarnoja 2018b.

## 1.6 CMDP: i vincoli come vincoli

Un CMDP aggiunge costi `c_i` e budget `b_i`:

```
max_π  J(π)    s.t.   C_i(π) = E[ Σ_t γ^t c_i(s_t,a_t) ] ≤ b_i
```

Risolto con il Lagrangiano:

```
max_π min_{λ≥0}  J(π) − λ ( C(π) − b )
```

L'attore massimizza `Q_r − λ·Q_c`; il moltiplicatore `λ` sale se il vincolo è violato e scende altrimenti.

**Perché ti conviene.** Nella formulazione scalarizzata devi scegliere `w_shortfall = 500 g/MJ`. Quel numero è arbitrario (§3.3) e in discussione ti verrà chiesto di giustificarlo. Nel CMDP scegli invece un **budget in MJ**, che è una specifica ingegneristica, non un peso. La domanda "perché 500?" diventa "perché 0.05 MJ?", che ha una risposta.

**Il dettaglio che si sbaglia sempre.** Il critico del costo usa `max` dei due, non `min`. Per il reward la stima pessimista è quella bassa; per il costo è quella alta. Usare `min` anche lì sottostima il costo e il vincolo non viene mai sentito.

### Domande sulla §1

1. Perché la proprietà di Markov è una scelta di progetto e non una proprietà da verificare?
2. Se togli `T_bat` dall'osservazione, cosa vede l'agente al posto del derating?
3. Perché non puoi fare controllo model-free con la sola `V(s)`?
4. Perché PPO è escluso dal tuo problema?
5. Cosa fa fisicamente il termine `−α log π` dentro il target di Bellman?
6. Perché serve la correzione jacobiana per il tanh, e cosa succederebbe se la omettessi?
7. Perché `min` sui critici del reward e `max` su quello del costo?
8. Il tuo problema ha 3886 step e orizzonte finito. Perché `γ = 0.995` e non `γ = 0.99`?

---

# 2. Architettura del codice

## 2.1 La struttura in una frase

L'ambiente incapsula il plant, il safety layer garantisce i vincoli hard per costruzione, il reward è espresso in grammi-equivalenti, e SAC vede un problema in cui deve solo ottimizzare — non scoprire cosa è ammissibile.

```
                    ┌──────────────┐
                    │   SAC agent  │  azione a ∈ [-1,1]
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  safety.py   │  [lb, ub] analitico
                    │  project()   │  → P2 fisico
                    └──────┬───────┘
                           │
          ┌────────────────▼────────────────┐
          │            env.py               │
          │  powertrain → battery_step      │
          │            → thermal_model      │
          │  reward in grammi-equivalenti   │
          └────────────────┬────────────────┘
                           │
                  obs, reward, cost, info
```

## 2.2 Ordine di lettura

| # | file | tempo | cosa cercare |
|---|---|---|---|
| 1 | `config.py` | 10 min | la mappa completa delle manopole |
| 2 | `safety.py` | 20 min | i cinque vincoli, uno per uno |
| 3 | `env.py` | 30 min | `_obs`, poi `step`, poi `summary` |
| 4 | `baselines.py` | 15 min | verifica: ritrovi ECMS come policy dell'env |
| 5 | `evaluate.py` | 10 min | `fuel_eq_g` |
| 6 | `sac.py` | 40 min | solo dopo l'ambiente |
| 7 | `buffer.py`, `bc.py`, `train.py` | 20 min | infrastruttura |

Le due funzioni che contano davvero sono `feasible_P2_bounds` ed `EMSEnv.step`. Circa 120 righe. Tutto il resto è sostituibile.

---

# 3. Walkthrough, file per file

## 3.1 `config.py`

Tre dataclass. Nessuna logica. La regola: **se una manopola non compare qui, non esiste**. Un esperimento è descritto interamente da `(EnvConfig, SACConfig, TrainConfig, seed)`, e `dump_configs` li serializza accanto ai risultati.

Valori che vale la pena aver presenti: `preview_horizon=5`, `preview_stride=5` (→ 2.5 s di orizzonte), `w_shortfall=500`, `w_rate=1.0`, `w_soc_lin=260`, `w_soc_quad=20000`, `reward_scale=0.05`, `gamma=0.995`, `buffer_size=1e6`, `cost_limit=0.05` MJ.

## 3.2 `safety.py` — il cuore

Una funzione, `feasible_P2_bounds`, che restituisce l'intervallo ammissibile `[lb, ub]` per la potenza elettrica. Cinque vincoli:

```python
ub = P_MGU_max * derate(T_bat)              # 1. inviluppo MGU-K, derato
ub = min(ub, Uoc²/(4·R_int))                # 2. solvibilità della quadratica
ub = min(ub, (E_deploy_max − E_acc)/dt)     # 3. budget deploy per giro
ub = min(ub, 0.98 · I_dis_max · U)          # 4a. SoC ≥ SoC_min
lb = max(P_MGU_min, −0.98 · I_chg_max · U)  # 4b. SoC ≤ SoC_max
lb = max(lb, P_gb · eta_MGU)  se P_gb < 0   # 5. frenata disponibile
```

**Cosa capire, non memorizzare.**

Il vincolo 2 non è un limite fisico: `Uoc²/(4R)` è il punto di massimo trasferimento di potenza, cioè metà OCV caduta su `R_int` — a 300 V sono 150 V di caduta e ~15 kA. Vale 1.8–3.0 MW. Serve solo perché oltre quel valore il discriminante di `U2² − Uoc·U2 + P2·R = 0` diventa negativo.

Il vincolo 4 usa `Uoc` al posto della tensione ai morsetti `U2`. In scarica `U2 < Uoc`, quindi a parità di potenza la corrente reale è **maggiore** di quella stimata: la stima è ottimista, da cui il margine 0.98.

Il vincolo 5 è quello che si dimentica: durante la frenata non puoi recuperare più potenza di quella che il cambio sta dissipando. `P_mech ≥ P_gb`, e in dominio elettrico `P2 ≥ P_gb · η_MGU`.

**La guardia finale.**

```python
if lb > ub:
    mid = 0.5 * (lb + ub)
    lb = ub = clip(mid, P_MGU_min, P_MGU_max)
```

Un intervallo invertito passato a `np.linspace` o a un campionatore produce output silenziosamente privo di senso. È lo stesso bug che avevi incontrato nel backward pass vettorizzato. Qui è impossibile per costruzione, e c'è un test che lo verifica su tutti gli angoli dello spazio di stato.

**`project` e `inverse_project`.** La mappatura `relative` manda `a=-1 → lb` e `a=+1 → ub`. L'inversa serve al behaviour cloning: per clonare l'ECMS devi convertire il suo `P2` nell'azione normalizzata che l'agente avrebbe dovuto emettere in quello stato.

### Domande sulla §3.2

9. Perché `Uoc²/(4R)` non è un limite di sicurezza della batteria?
10. Perché il margine è 0.98 e non 1.0, e perché serve solo in scarica?
11. Cosa impedisce il vincolo 5, e cosa succederebbe senza?
12. Con la mappatura `relative`, `a = 0.5` significa la stessa cosa in due stati diversi?
13. Perché `inverse_project` esiste, se l'agente emette solo azioni in avanti?

## 3.3 `env.py`

**`_obs` — 8 + `preview_horizon` componenti.**

```
[SoC, T_bat, E_deploy residuo, P_gb, v, a, posizione_giro, giri_rimanenti, preview...]
```

Normalizzazione analitica fissa, nessuna running statistic: un checkpoint è riproducibile senza stato aggiuntivo.

Le ultime due componenti prima della preview sono quelle che si dimenticano. Il problema è a **orizzonte finito e non stazionario**: la politica ottima dipende da quanta gara resta. È letteralmente il motivo per cui il tuo DP multi-lap senza vincolo svuota il pacco nei primi due giri — ha ragione a farlo, se nessuno gli dice che la gara continua.

**`step` — cinque blocchi numerati.**

1. **azione → comando fisico.** Con `safety_layer=True` si proietta; con `False` si clippa e si segnala `projected=True`, così puoi misurare quanto spesso l'agente chiederebbe l'impossibile.
2. **plant.** `powertrain(..., control_mode="P2", Tbat_k=T_bat)` → `battery_step` → `thermal_model`. Dopo il fix termico, `Tbat_k` è ciò che fa arrivare il vincolo termico al bilancio di potenza.
3. **reward.** Vedi sotto.
4. **confine di giro.** Hinge asimmetrica sul SoC, reset del budget di deploy.
5. **bookkeeping.**

C'è anche un controllo che vale la pena notare:

```python
silent_clip = c.safety_layer and abs(P2 - P2_cmd) > 1e-3
```

Se il plant clippa dietro le spalle del controllore, richiesto ed erogato divergono e lo `shortfall` restituito da `powertrain()` smette di essere la fonte unica di verità. C'è un test che verifica che non succeda mai su un episodio intero.

**Il reward, in grammi-equivalenti.**

```
r = −( fuel_g + w_shortfall·unmet_MJ + w_rate·|ΔP2|_MW ) · scale
r_fine_giro −= w_soc_lin·h + w_soc_quad·h²,   h = max(0, SoC_target − SoC)
```

Le derivazioni, che devi saper rifare:

- 1 MJ non erogato ≡ `1e6/(η_ICE·LHV) = 1e6/(0.498·44e6) = 45.6 g`. Prezzo naturale.
- 1 unità di SoC ≡ `E_pack/(η_ICE·LHV)·1000 = 5.7e6/(0.498·44e6)·1000 = 260 g`.
- `w_shortfall = 500` è ~10× il prezzo naturale. **È l'unico numero arbitrario del reward**, e il motivo è che uno shortfall è una perdita di tempo sul giro, non una spesa di carburante. Nel CMDP sparisce.

Il punto non negoziabile: la penalità di shortfall è **continua** nell'entità. Un `BIG_M` binario rende l'ottimizzatore indifferente tra 0.1 MJ e 12 MJ. In DP costava un'approssimazione; in RL è fatale, perché il gradiente è l'unico segnale.

**`summary` e `fuel_eq_g`.**

```
fuel_eq = fuel + (SoC_target − SoC_finale) · 260
```

È la cifra da mettere in tabella. Confrontare carburante grezzo tra controllori che finiscono a SoC diverso non è un confronto: chiunque sembra frugale arrivando a pacco vuoto. È esattamente il motivo per cui il `−11.5%` del rule-based contro il DP non era difendibile.

### Domande sulla §3.3

14. Perché `laps_remaining` nell'osservazione, e cosa cambia se lo togli?
15. Perché la normalizzazione è fissa e non adattiva?
16. Rifai a mano il calcolo di 45.6 g/MJ e di 260 g per unità di SoC.
17. Cosa succede al training se `w_shortfall` diventa binario?
18. Perché `reward_scale = 0.05` non ha alcun contenuto fisico?
19. Se un controllore finisce a `SoC = 0.2` invece di `0.8`, quanti grammi gli aggiunge `fuel_eq_g`?
20. Cosa segnala `silent_clip = True`, e perché sarebbe grave?

## 3.4 `baselines.py`

Rule-based, ECMS e replay-DP come policy dello stesso ambiente. La ragione è metodologica: i tre controllori esistono già nei notebook, ma ognuno gira nel proprio loop. Un confronto costruito su tre loop diversi è vulnerabile all'obiezione che la differenza venga dall'harness. Qui: stesso plant, stesso safety layer, stessa contabilità.

L'ECMS calcola `s(t) = s₀ − Kp·(SoC − SoC_ref)` e minimizza l'Hamiltoniana su una griglia dell'intervallo ammissibile. Griglia e non forma chiusa, perché il costo non è liscio (saturazione fuel-flow, budget, tetto termico) proprio nei punti che contano.

## 3.5 `sac.py`

Mappa mentale: `SquashedGaussianActor` (corpo + testa media + testa log-std), `Critic` (due `Q` in un modulo solo), `SAC` che li orchestra.

In `update`, l'ordine: critico → cost critic (se CMDP) → attore (delayed) → temperatura → target network. La sequenza non è arbitraria: l'attore usa i `Q` appena aggiornati, e `alpha` usa la `logp` calcolata nel blocco dell'attore.

Due dettagli da notare:

```python
for p in self.critic.parameters():
    p.requires_grad_(False)
```

Durante l'update dell'attore i gradienti non devono fluire nel critico. Alternativa comune è un `.detach()`, ma qui il grafo passa per `a_pi` che deve restare derivabile — quindi si congelano i parametri, non il tensore.

```python
self.ep_cost_est = 0.9*self.ep_cost_est + 0.1*episode_cost
```

`λ` si aggiorna **una volta per episodio**, non per step. Il vincolo è un budget sull'intera gara: un moltiplicatore che reagisce a una quantità per-step insegue rumore.

### Domande sulla §3.5

21. Perché due critici e non uno, e perché il `min`?
22. Perché `requires_grad_(False)` e non `.detach()`?
23. Cosa succede se `alpha` diverge verso l'alto?
24. Perché `update_lambda` è chiamata per episodio e non dentro `update`?
25. `policy_delay = 2` trasformerebbe SAC in TD3? Cosa mancherebbe ancora?

## 3.6 `buffer.py`, `bc.py`, `train.py`

**Buffer.** Canale di costo separato (serve al CMDP) e prioritizzazione opzionale. La prioritizzazione conta perché gli episodi sono lunghi e le transizioni interessanti sono rare: esaurimento budget, derating attivo, domanda non soddisfatta. Il campionamento uniforme le annega.

**BC.** Supervisiona solo la testa della media, non la log-std: vuoi un attore che sappia *dove* agire ma resti libero di esplorare *quanto*.

**Train.** Curriculum 1 → 3 → 5 giri, domain randomization su SoC iniziale, temperatura e `cooling_factor`. Quest'ultima ha un secondo scopo oltre all'anti-overfitting: garantire che il vincolo termico venga **esercitato**.

Una riga da capire:

```python
buf.add(obs, a, r, nobs, float(term), cost=info["cost"])
```

`term`, non `term or trunc`. La troncatura a fine profilo è un limite di tempo artificiale: bootstrappare attraverso di essa è corretto, trattarla come terminale falsa sistematicamente la value function verso il basso in prossimità della fine dell'episodio. È un errore comune e silenzioso.

### Domande sulla §3.6

26. Perché il BC non supervisiona la log-std?
27. Perché `float(term)` e non `float(term or trunc)`?
28. Perché randomizzare il `cooling_factor` invece di addestrare a raffreddamento nominale?
29. In che senso il canale di costo del buffer è "separato" dal reward?

---

# 4. Esercizi guidati

## 4.1 Strumenta un singolo step

Dieci minuti, e ti insegna `safety.py` ed `env.step` meglio di un'ora di lettura.

```python
from rl.config import EnvConfig
from rl.env import EMSEnv

env = EMSEnv(EnvConfig(profile='single_lap'))
obs, _ = env.reset(seed=0)

for i in range(300):
    lb, ub = env._bounds()
    obs, r, term, trunc, info = env.step([0.8])
    if i % 50 == 0:
        print(f"k={i:4d}  P_gb={env.prof['P_gb'][i]/1e3:7.1f} kW  "
              f"env=[{lb/1e3:7.1f},{ub/1e3:7.1f}]  P2={info['P2']/1e3:7.1f}  "
              f"derate={info['derate']:.2f}  SoC={info['SoC']:.3f}  "
              f"T={info['T_bat']:.1f}  fuel={info['fuel_g']:.3f}  r={r:.3f}")
```

Da osservare:
- l'inviluppo si stringe man mano che il budget di deploy si consuma;
- si stringe di nuovo quando `T_bat` supera 45 °C;
- `lb` è negativo solo quando `P_gb < 0`.

Poi verifica a mano che `r == −(fuel_g + 500·shortfall_MJ + rate)·0.05`.

## 4.2 Rompi un test di proposito

In `safety.py`, commenta il vincolo 3:

```python
# ub = min(ub, e_left / dt)
```

Poi `pytest tests -q`. Guarda quali test cadono e leggi i loro docstring. Ripeti con il vincolo 1 (derating) e con la guardia `if lb > ub`. Ripristina con `git checkout rl/safety.py`.

Ogni test è la specifica di una decisione. Rompendola scopri cosa proteggeva — è il modo più veloce per capire un sistema che non hai scritto tu.

## 4.3 Riproduci i numeri

Baseline giro singolo:

| controller | fuel_eq [g] | fuel [g] | short [MJ] | therm | SoC_f |
|---|---|---|---|---|---|
| rule_based | 843.56 | 777.07 | 1.9237 | 97 | 0.544 |
| ecms | 895.72 | 853.06 | 1.0589 | 229 | 0.636 |

ECMS a cinque giri sotto stress termico:

| cooling | fuel_eq [g] | short [MJ] | therm | SoC_f |
|---|---|---|---|---|
| 1.0 | 4154.97 | 2.1130 | 882 | 0.675 |
| 0.5 | 4103.66 | 3.9215 | 1265 | 0.702 |
| 0.3 | 4055.69 | 5.9470 | 1614 | 0.765 |

Se li ottieni, hai capito come si incastrano ambiente, safety layer e plant termico. Se non tornano, il punto in cui divergono ti dice cosa non hai capito.

Nota la lettura della seconda tabella: il carburante *scende* col raffreddamento peggiore. È un artefatto, ed è esattamente ciò che `fuel_eq` esiste per smascherare — guarda sempre carburante equalizzato **e** shortfall insieme.

## 4.4 Ablazioni che valgono un paragrafo di tesi

| ablazione | come | cosa misuri |
|---|---|---|
| preview 0 | `EnvConfig(preview_horizon=0)` | quanto del vantaggio DP è informazione |
| niente safety layer | `safety_layer=False` | quanti step l'agente chiede l'impossibile |
| `w_rate = 0` | `EnvConfig(w_rate=0)` | il chattering è strutturale |
| azione assoluta | `action_mode='absolute'` | costo della mappatura a semantica fissa |
| CMDP vs scalare | `--lagrangian` | se il budget batte il peso a mano |
| niente BC | `--no-bc` | valore del warm-start |

---

# 5. Diagnosi dei guasti

I sintomi che vedrai, e cosa significano.

| sintomo | causa probabile | dove guardare |
|---|---|---|
| shortfall **cresce** durante il training | critico casuale che distrugge l'attore clonato | `actor_start_it`, warmup |
| policy torna casuale dopo un buon inizio | `alpha` diverge | logga `info["alpha"]`, `target_entropy` |
| l'agente satura sempre a `a = ±1` | reward mal scalato o `Q` che esplode | `reward_scale`, `loss_q` |
| oscillazione ad alta frequenza in `P2` | `w_rate` troppo basso | confronta `w_rate = 0` e `w_rate = 5` |
| il SoC finisce sempre a `SoC_min` | penalità di fine giro troppo debole | `w_soc_quad` |
| risultati non riproducibili | seed non propagato, o randomizzazione attiva in eval | `test_determinism_given_a_seed` |
| `fuel` scende ma `fuel_eq` sale | l'agente sta "vincendo" scaricando il pacco | è la metrica che funziona |
| vincolo termico mai attivo | `Tbat_k` non passato, o raffreddamento troppo buono | `cooling_factor_range` |

**Il caso reale che hai incontrato.** Nel run a 20k step lo shortfall è andato da 1.41 MJ (attore clonato) a 6.07 MJ. Due bug concorrenti: il warmup campionava azioni casuali ignorando il checkpoint BC, e il critico partiva casuale spingendo l'attore a massimizzare una `Q` priva di significato. È catastrophic forgetting da manuale, e il segnale diagnostico era che il **reward** peggiorava — non solo la metrica.

Regola generale: se il reward di training peggiora monotonamente, non è un problema di iperparametri, è un bug.

---

# 6. Osservazioni sul plant che meritano attenzione

Emerse leggendo `parameters.py` mentre scrivevo il layer RL. Non sono errori certi, sono cose da verificare.

**`T_coolant_in = 50 °C` contro `T_bat_derate_start = 45 °C`.** Il refrigerante è più caldo della soglia di derating. A regime, senza carico, la batteria tende a 50 °C, cioè stabilmente dentro la zona derata con `derate ≈ 2/3`. E `T_bat_init = 40 °C` significa che il pacco viene *scaldato* dal circuito, non raffreddato. Questo spiega i conteggi di binding events alti che vedi. In F1 il refrigerante ERS sta tipicamente sotto i 40 °C. Se è intenzionale va giustificato in tesi; se è un refuso, cambia tutti i risultati termici.

**`E_deploy_max = 9 MJ` per giro contro `E_pack` usabile di 4 MJ.** Coerente se prevedi ricarica in-lap, che è il caso, ma verifica il numero regolamentare 2026 che vuoi citare.

**`P_ICE_max = 400 kW` e `m_dot_max = 68.18 kg/h`.** Il vincolo di fuel flow dà `η_ICE · LHV · ṁ − P_ICE0 = 0.498·44e6·0.01894 − 15e3 ≈ 400 kW`. Torna, ed è la calibrazione che il tuo README dichiara. Buon segno.

---

# 7. Tutte le domande

## Livello 1 — hai letto

1. Perché la proprietà di Markov è una scelta di progetto?
2. Cosa vede l'agente al posto del derating, se togli `T_bat`?
3. Perché non basta `V(s)` per il controllo model-free?
4. Perché PPO è escluso?
5. Cosa fa `−α log π` nel target di Bellman?
6. Perché la correzione jacobiana del tanh?
7. Perché `min` sul reward e `max` sul costo?
8. Perché `γ = 0.995`?
9. Perché `Uoc²/(4R)` non è un limite di sicurezza?
10. Perché il margine 0.98, e solo in scarica?
11. Cosa impedisce il vincolo 5?
12. Con `relative`, `a = 0.5` significa la stessa cosa in due stati diversi?
13. Perché esiste `inverse_project`?
14. Perché `laps_remaining` nell'osservazione?
15. Perché normalizzazione fissa e non adattiva?
16. Rifai il calcolo di 45.6 g/MJ e 260 g/SoC.
17. Cosa succede se `w_shortfall` diventa binario?
18. Perché `reward_scale` non ha contenuto fisico?
19. Quanti grammi aggiunge `fuel_eq_g` a un controllore che finisce a `SoC = 0.2` invece di `0.8`?
20. Cosa segnala `silent_clip = True`?
21. Perché due critici e il `min`?
22. Perché `requires_grad_(False)` e non `.detach()`?
23. Cosa succede se `alpha` diverge?
24. Perché `update_lambda` per episodio?
25. `policy_delay = 2` basta per avere TD3?
26. Perché il BC non supervisiona la log-std?
27. Perché `float(term)` e non `float(term or trunc)`?
28. Perché randomizzare il `cooling_factor`?
29. In che senso il canale di costo è separato?

## Livello 2 — hai capito

30. Perché l'azione è `P2` e non `u_split`?
31. Perché il DP non può essere battuto, e perché non è un problema?
32. Come separi l'errore di approssimazione dalla perdita di informazione?
33. Perché il chattering ECMS è strutturale e non un bug?
34. Perché il rule-based consuma meno del DP e la cosa non significa niente?
35. Cosa misura esattamente il gap RL–DP in percentuale?
36. Perché il safety layer rende il problema *più facile* da imparare, non solo più sicuro?
37. Nel CMDP, cosa sostituisce la domanda "perché 500 g/MJ?"
38. Perché lo stress test sul raffreddamento è una distribuzione di task e non solo un'analisi di sensitività?
39. Perché il carburante scende con raffreddamento peggiore, e perché non è una buona notizia?
40. Perché il valore della griglia DP è una stima del control invariant set?

## Livello 3 — sapresti estenderlo

41. Come implementeresti l'RL che tara `s(t)` dell'ECMS invece di comandare `P2`? Cosa cambia nell'ambiente?
42. Come costruiresti una control barrier function sul vincolo termico, sapendo che hai `ḣ` in forma chiusa?
43. Come useresti il value grid del DP per il campionamento degli stati iniziali?
44. Cosa cambierebbe passando a Distributional SAC, e perché il tuo rischio è asimmetrico?
45. Come formuleresti la versione POMDP, e quale architettura di rete servirebbe?
46. Come vettorizzeresti l'ambiente per N istanze parallele? Qual è il collo di bottiglia?
47. Come costruiresti un benchmark meta-RL con "un task = un circuito"?

---

# 8. Risposte

Usale dopo aver provato.

**1.** Perché lo stato lo scegli tu. Il plant è deterministico; se ometti una variabile di stato, l'agente osserva transizioni che sembrano stocastiche ma sono deterministiche in una variabile nascosta.

**2.** Rumore. Il derating avviene, ma l'agente non ha modo di prevederlo, quindi lo interpreta come varianza dell'ambiente.

**3.** Per scegliere l'azione da `V` devi sapere dove ti porta ciascuna azione, cioè ti serve il modello. `Q` ha già incorporato quel passo.

**4.** È on-policy: butta i dati dopo ogni update. Con 3886 valutazioni del plant per episodio è insostenibile.

**5.** Premia l'incertezza. Rende l'esplorazione parte dell'obiettivo invece di un'aggiunta euristica, il che conta perché il tuo paesaggio di costo ha discontinuità.

**6.** Perché `tanh` cambia la densità. Senza la correzione, `log π` è sbagliata, il termine entropico è mal calcolato e `alpha` si autoregola su una quantità priva di senso.

**7.** Sul reward la stima pessimista è quella bassa (limita la sovrastima); sul costo è quella alta (evita di sottostimare il vincolo).

**8.** Orizzonte finito: `γ` non serve alla convergenza, solo al credit assignment. Il tuo obiettivo è una somma non scontata, quindi `γ` vicino a 1 mantiene il problema che il DP risolve. `0.99` su 3886 step sconta il futuro troppo aggressivamente e cambia il problema.

**9.** È il punto di massimo trasferimento di potenza del circuito equivalente: metà OCV su `R_int`, ~15 kA a 300 V. Nessun pacco ci arriva. È un limite di solvibilità matematica.

**10.** Perché la stima usa `Uoc` invece di `U2`. In scarica `U2 < Uoc`, quindi a parità di potenza la corrente reale è maggiore: la stima è ottimista e serve margine. In carica l'errore ha segno opposto.

**11.** Che si recuperi più potenza di quella che il cambio sta dissipando. Senza, l'agente genererebbe energia dal nulla in frenata.

**12.** No. Significa "metà dell'inviluppo disponibile", e l'inviluppo cambia con lo stato. È il trade-off dichiarato della mappatura `relative`.

**13.** Per il behaviour cloning: devi convertire il `P2` del maestro nell'azione normalizzata che l'agente avrebbe dovuto emettere in quello stato.

**14.** Perché il problema è a orizzonte finito e non stazionario. Senza, l'agente non può sapere se conviene scaricare o conservare, ed è il motivo per cui il DP multi-lap senza vincolo svuota il pacco subito.

**15.** Riproducibilità: un checkpoint funziona senza portarsi dietro statistiche accumulate. Costo: se cambi circuito devi verificare le costanti di scala.

**16.** `1e6/(0.498·44e6) = 0.0456 kg = 45.6 g`. `5.7e6/(0.498·44e6)·1000 = 260 g`.

**17.** Il gradiente rispetto all'entità dello shortfall sparisce. L'agente diventa indifferente tra 0.1 e 12 MJ e non ha modo di migliorare gradualmente.

**18.** È solo condizionamento numerico per la rete. Moltiplicare tutti i reward per una costante positiva non cambia la politica ottima, cambia la scala di `Q` e quindi dei gradienti.

**19.** `(0.8 − 0.2) · 260 = 156 g`.

**20.** Che il plant ha clippato dietro le spalle del controllore: richiesto ed erogato divergono e lo shortfall smette di essere la fonte unica di verità del bilancio di potenza.

**21.** L'errore di approssimazione è rumore a media circa nulla, ma il `max` dell'operatore di Bellman lo converte sistematicamente in sovrastima, che si accumula. Il `min` di due stime indipendenti la contrasta.

**22.** Perché il grafo deve restare derivabile attraverso `a_pi` (serve il gradiente dell'attore), ma non deve accumulare gradienti nei parametri del critico. `.detach()` su `q_pi` taglierebbe anche il percorso verso l'attore.

**23.** La politica torna verso il rumore: il termine entropico domina il reward. Sintomo tipico con azione 1-D, dove `target_entropy = −1` è già molto permissivo.

**24.** Perché il vincolo è un budget sull'intera gara. Un moltiplicatore che reagisce a un costo per-step insegue rumore invece della violazione effettiva.

**25.** No. Mancherebbero la politica deterministica e il target policy smoothing. `policy_delay` è solo uno dei tre ingredienti di TD3.

**26.** Perché vuoi un attore sicuro su *dove* agire ma libero di esplorare *quanto*. Supervisionando la std ottieni una politica quasi deterministica che annulla il termine entropico nelle prime migliaia di update.

**27.** La troncatura a fine profilo è un limite di tempo artificiale, non una terminazione. Trattarla come terminale dice al critico che il valore dopo l'ultimo step è zero, il che falsa la value function verso il basso vicino alla fine.

**28.** Perché a raffreddamento nominale il derating vincola raramente. Un agente addestrato solo lì non incontra mai il vincolo termico nel replay buffer e non impara a gestirlo.

**29.** Il costo ha il suo critico, il suo target e il suo moltiplicatore. Non entra nel reward: è un vincolo, non un termine dell'obiettivo. Nella modalità CMDP lo script azzera `w_shortfall` proprio per non prezzarlo due volte.

**30.** Perché il dominio elettrico è dove sono definiti sia il limite regolamentare di 350 kW sia il budget di deploy. È la stessa correzione che avevi applicato a `MGU_K()`: comandare `u_split` reintrodurrebbe l'errore per un'altra strada.

**31.** Perché il DP conosce l'intero futuro. Non è un problema: il DP è il limite superiore, e la metrica è la distanza da esso a parità di vincoli, con il vantaggio dell'inferenza in microsecondi.

**32.** Con l'ablazione della preview. Con preview il divario informativo è quasi chiuso e il gap misura l'approssimazione; senza, misura entrambe. La differenza tra i due gap è la componente informativa.

**33.** Perché nasce dall'assenza di un termine di penalità sulla rate nell'Hamiltoniana, non da un errore di implementazione. È distinto dall'esaurimento del budget MGU-K, che è una discontinuità fisica.

**34.** Perché lascia domanda non soddisfatta e carica inutilizzata. A SoC equalizzato consuma **di più** (843.56 contro 895.72 dell'ECMS a fronte di 1.92 MJ di shortfall contro 1.06).

**35.** L'errore di approssimazione di funzione, a parità di informazione, se la preview è attiva. Altrimenti la somma di approssimazione e osservabilità parziale.

**36.** Perché elimina dallo spazio di ricerca le regioni infattibili. L'agente non spende campioni a scoprire vincoli che tu conosci in forma chiusa, e ogni valore dell'azione resta significativo.

**37.** "Perché un budget di 0.05 MJ?", che è una specifica ingegneristica con una risposta, invece di un peso relativo che non ne ha.

**38.** Perché con `cooling_factor_range` ogni episodio campiona un ambiente diverso: è domain randomization, e la politica appresa è robusta alla degradazione invece che ottimizzata per un punto.

**39.** Perché il derating impedisce di usare l'elettrico, il pacco non si scarica e il SoC finale resta alto: il carburante grezzo scende ma l'energia non è stata spesa. `fuel_eq` lo corregge e lo shortfall raddoppia.

**40.** Perché gli stati con valore finito sono, per costruzione, quelli da cui esiste una politica che completa la gara senza violare i vincoli. È la definizione di insieme controllo-invariante, discretizzata.

**41.** L'azione diventa `Δs`, e `step` chiama l'ECMS con l'`s` aggiornato invece di proiettare direttamente. Lo stato guadagna `s` corrente; l'inviluppo lo garantisce comunque il solutore ECMS. Il resto — reward, metriche, harness — resta identico.

**42.** `h(x) = T_derate − T_bat`, con `ḣ = −dT/dt` dal modello a capacità concentrata, che è affine in `I²R`. Il vincolo `ḣ ≥ −α h` diventa un vincolo quadratico su `I2`, quindi su `P2`: un QP a una variabile risolvibile in forma chiusa a ogni step.

**43.** Campionando lo stato iniziale solo tra gli stati con valore finito, e resettando l'episodio quando la traiettoria ne esce. Sono due dei tre usi offline del CIS in Bo et al.; il terzo è la penalità nel reward.

**44.** Modelleresti la distribuzione dei ritorni invece della media, potendo ottimizzare quantili. Il tuo rischio è asimmetrico perché uno shortfall è inaccettabile mentre un eccesso di margine costa solo carburante: ottimizzare il 95° percentile è più vicino a ciò che vuoi della media.

**45.** `preview_horizon = 0` e una politica ricorrente (GRU o LSTM) su attore e critico, con il buffer che memorizza sequenze invece di transizioni singole.

**46.** Riscrivendo `step` per operare su array di N stati invece che su scalari. Il collo di bottiglia è il loop Python dentro `powertrain` e `battery_step`: la versione vettorizzata richiede di riscrivere quelle due funzioni con `np.where` al posto dei rami condizionali.

**47.** Un task = un circuito, con FastF1 che fornisce le tracce. Meta-addestri su un sottoinsieme e misuri quanti giri servono per adattarsi a un circuito mai visto. È la narrativa perfetta per il motorsport: 24 circuiti, stesso hardware.

---

# 9. Cosa leggere dopo

In ordine di priorità, e solo le fonti primarie — la review di Devarakonda citala per inquadramento, mai come fonte di un metodo.

1. **Haarnoja et al. 2018**, SAC — obbligatorio. Poi 2018b per l'auto-tuning di `α`, che nella review non c'è.
2. **Sutton & Barto**, 2ª ed. — cita questo per MDP e Bellman, non la review.
3. **Fujimoto et al. 2018**, TD3 — baseline di confronto obbligatorio.
4. **Achiam et al. 2017**, CPO — se vai su CMDP.
5. **Bo et al. 2023**, CIS-enhanced safe RL — il collegamento value grid DP → control invariant set.
6. **Brunke et al.**, Safe Learning in Robotics — i tre livelli di sicurezza, per classificare i tuoi vincoli.
7. **Ames et al. 2017**, CBF-based QP — se implementi la barriera termica.
8. **Schaul et al. 2016**, Prioritized Experience Replay.
