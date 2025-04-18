#include <math.h>

#define IDIM 300

int IR;
int NI, NIM; 
int i,j,k;
int iter, ITMAX;
double ERRMAX;

double Tsolid, Tgas;
double L, DX0, MDOT, Utot, TGIN0, TSIN0;
double XP[IDIM], XU[IDIM], DXP[IDIM], DXU[IDIM];
double BTG[IDIM], BTS[IDIM];
double TG[IDIM], TGSOR[IDIM],TG_old[IDIM], TS[IDIM], TSSOR[IDIM], TS_old[IDIM];
double KG, CPG, CPS, KS, QDOT, TKKP[IDIM];

double CC[IDIM], SS[IDIM];
double AP[IDIM], AW[IDIM], AE[IDIM];
//double APS[IDIM], AWS[IDIM], AES[IDIM];

void TDMA(int ist, int iend, double T[IDIM]);
void INIT();
void TG_SOLVE();
void TS_SOLVE();
double TKK_INTP(double a, double b);


