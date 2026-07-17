from .parameters import params

def aero_drag(v, params):
    rho_a = params['rho_a']  # [kg/m^3]density of the ambient air
    Af = params['Af'] #[m] frontal area
    cd = params['Cd'] #[-] aerodynamic drag coeﬃcient
    Fa = 0.5 * rho_a * Af * cd * v**2
    
    return Fa 
    

def rolling_resistance(params): 
    cr = params['Cr'] #[-] rolling friction coeﬃcient
    mv = params['mv'] #[kg] vehicle mass
    g = params['g'] # [m/s^2] acceleration gravity
    Fr = cr * mv * g #suppose horizontal road (alpha = 0)
    
    return Fr 


def longitudinal_dynamics(v, a, params):
    F_rolling = rolling_resistance(params)
    F_aero = aero_drag(v, params)
    mv = params['mv']
    eta_gearbox = params['eta_gearbox']

    Fx = (mv * a) + F_rolling + F_aero
    Pm = Fx * v # power required
    if Pm > 0:
        P_gb = Pm / eta_gearbox
    else:
        P_gb = Pm * eta_gearbox


    return Fx, Pm, P_gb

