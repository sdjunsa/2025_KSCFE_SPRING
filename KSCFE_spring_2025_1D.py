import numpy as np

# -------------------------------------------------------------------
# 완전 구현된 Python 버전 of 1D Heat Exchanger
# -------------------------------------------------------------------

# 시뮬레이션 파라미터
t_end = 10.0
dt = 0.1
n_time_steps = int(t_end / dt)
rhog = 1.0  # 가스 밀도
rhos = 1.0  # 고체 밀도

# 물성치 및 경계조건
L = 1.0
NI = 80
DX0 = L / (NI - 2)
KG = 0.1      # 가스 열전도도
KS = 1.6      # 고체 열전도도
CPG = 1000.0  # 가스 비열
CPS = CPG     # 고체 비열 (원 코드와 동일)
Utot = 50.0   # 전체 열전달 계수
TGIN0 = 1400.0
TSIN0 = 300.0
MDOT = 2.6    # 질량 유량 [kg/m2s]
ITMAX = 100
ERRMAX = 0.01

# 배열 할당 (1-based 인덱싱, 인덱스 0 무시)
XP      = np.zeros(NI+1)
XU      = np.zeros(NI+1)
DXP     = np.zeros(NI+1)
DXU     = np.zeros(NI+1)
BTG     = np.zeros(NI+1)
BTS     = np.zeros(NI+1)
TG      = np.zeros(NI+1)
TG_old  = np.zeros(NI+1)
TS      = np.zeros(NI+1)
TS_old  = np.zeros(NI+1)
TGSOR   = np.zeros(NI+1)
TSSOR   = np.zeros(NI+1)
AW      = np.zeros(NI+1)
AE      = np.zeros(NI+1)
AP      = np.zeros(NI+1)
SS      = np.zeros(NI+1)

def INIT():
    global NIM
    NIM = NI - 1
    for i in range(2, NI+1):
        XU[i] = DX0 * (i - 2)
    for i in range(2, NIM+1):
        XP[i] = 0.5 * (XU[i+1] + XU[i])
    XP[1] = XU[2]
    XP[NI] = XU[NI]
    for i in range(2, NIM+1):
        DXP[i] = XU[i+1] - XU[i]
    for i in range(2, NI+1):
        DXU[i] = XP[i] - XP[i-1]
    for i in range(2, NIM+1):
        TG[i] = TGIN0
        TS[i] = TSIN0
    TG[1], TG[NI] = TGIN0, 600.0
    TS[1], TS[NI] = TSIN0, TSIN0

def TDMA(ist, iend, x):
    gamma = np.zeros_like(AP)
    beta = AP[ist]
    x[ist] = SS[ist] / beta
    for i in range(ist+1, iend+1):
        gamma[i] = -AE[i-1] / beta
        beta = AP[i] + AW[i] * gamma[i]
        x[i] = (SS[i] + AW[i] * x[i-1]) / beta
    for i in range(iend-1, ist-1, -1):
        x[i] -= gamma[i+1] * x[i+1]

def TG_SOLVE(iteration):
    for i in range(2, NIM+1):
        AW[i] = KG / DXU[i] + MDOT * CPG / DXU[i]
        AE[i] = KG / DXU[i+1] if i < NIM else 0.0
        AP[i] = AW[i] + AE[i] + Utot / DXP[i] + (rhog * DXP[i] * CPG / dt)
        Tsolid = BTS[i] if iteration > 0 else TS[i]
        TGSOR[i] = Utot / DXP[i] * Tsolid + (rhog * DXP[i] * CPG / dt) * TG_old[i]
        SS[i] = TGSOR[i]
    SS[2]   += AW[2] * TG[1]
    SS[NIM] += AE[NIM] * TG[NI]
    TDMA(2, NIM, TG)
    TG[NI] = TG[NIM]

def TS_SOLVE(iteration):
    for i in range(2, NIM+1):
        AW[i] = KS / DXU[i]
        AE[i] = KS / DXU[i+1] if i < NIM else 0.0
        if i == 2:
            AW[i] = 0.0
        Tgas = BTG[i] if iteration > 0 else TG[i]
        AP[i] = AW[i] + AE[i] + Utot / DXP[i] + (rhos * DXP[i] * CPG / dt)
        TSSOR[i] = Utot / DXP[i] * Tgas + (rhos * DXP[i] * CPG / dt) * TS_old[i]
        SS[i] = TSSOR[i]
    SS[2]   += AW[2] * TS[1]
    SS[NIM] += AE[NIM] * TS[NI]
    TDMA(2, NIM, TS)
    TS[1], TS[NI] = TS[2], TS[NIM]

if __name__ == "__main__":
    INIT()
    TG_old[:] = TG[:]
    TS_old[:] = TS[:]

    for t_step in range(n_time_steps):
        t = t_step * dt
        print(f"Time step {t_step}, t = {t:.6f} s")

        for iteration in range(ITMAX):
            BTG[2:NIM+1] = TG[2:NIM+1]
            BTS[2:NIM+1] = TS[2:NIM+1]

            TG_SOLVE(iteration)
            gsum = np.sum((TGSOR[2:NIM+1] + AW[2:NIM+1]*TG[1:NIM] + AE[2:NIM+1]*TG[3:NI+1] - AP[2:NIM+1]*TG[2:NIM+1])**2)
            gerr = np.sqrt(gsum / (NI-2))
            print(f" iter = {iteration:2d}, gerr = {gerr:15.7e}")

            TS_SOLVE(iteration)
            ssum = np.sum((TSSOR[2:NIM+1] + AW[2:NIM+1]*TS[1:NIM] + AE[2:NIM+1]*TS[3:NI+1] - AP[2:NIM+1]*TS[2:NIM+1])**2)
            serr = np.sqrt(ssum / (NI-2))
            print(f" iter = {iteration:2d}, serr = {serr:15.7e}")

            if gerr < ERRMAX and serr < ERRMAX:
                break

        TG_old[:] = TG[:]
        TS_old[:] = TS[:]

        if abs(t % 0.1) < 1e-8:
            fname_g = F"/1D_python/outg_{t:.1f}.txt"
            fname_s = F"/1D_python/outs_{t:.1f}.txt"
            with open(fname_g, "w") as fg:
                for i in range(1, NI+1):
                    fg.write(f"{i:4d} {XP[i]:15.7e} {TG[i]:15.7e}\n")
            with open(fname_s, "w") as fs:
                for i in range(1, NI+1):
                    fs.write(f"{i:4d} {XP[i]:15.7e} {TS[i]:15.7e}\n")
            print(f"Saved data at t = {t:.6f} s")

    print("Simulation complete.")
