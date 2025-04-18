// 1DHeatExchanger.cpp: 콘솔 응용 프로그램의 진입점을 정의합니다.
//
// 220210022Ex1.2.cpp: 콘솔 응용 프로그램의 진입점을 정의합니다.
//

#include "stdafx.h"
#include <iostream>
#include "common.h"

// 시간 관련 전역 변수
double t_end = 10.0;   
double dt = 0.1;      
int n_time_steps = t_end / dt;

double rhog = 1.0;
double rhos = 1.0;
int main()
{
	L = 1; NI = 80; DX0 = L / (NI - 2);
	KG = 0.1; KS = 1.6; CPG = 1000.; Utot = 50.; TGIN0 = 1400.; TSIN0 = 300.;
	MDOT = 2.6; // [kg/m2s]
	TG[1] = 1440.; TG[NI] = 600.;

	ITMAX = 100; ERRMAX = 0.01;

	INIT();

	// 초기 조건 저장 (초기 온도 분포)
	double TG_old[IDIM], TS_old[IDIM];
	for (i = 1; i <= NI; i++) {
		TG_old[i] = TG[i];
		TS_old[i] = TS[i];
	}

	// 시간 루프: 각 시간 단계마다 새 해 구하기
	for (int t_step = 0; t_step < n_time_steps; t_step++) {
		double t = t_step * dt;
		printf("Time step %d, t = %f s\n", t_step, t);

		for (iter = 0; iter < ITMAX; iter++)
		{
			for (i = 1; i < NI; i++) {
				BTG[i] = TG[i];
				BTS[i] = TS[i];
			}

			TG_SOLVE();
			double gerr, gsum;
			gsum = 0.;
			for (i = 2; i <= NIM; i++) {
				gerr = TGSOR[i] + AW[i] * TG[i - 1] + AE[i] * TG[i + 1] - AP[i] * TG[i];
				gsum = gsum + pow(gerr, 2.);
			}
			gerr = sqrt(gsum / (NI - 2));
			printf(" iter = %2d, gerr = %15.7e \n", iter, gerr);

			TS_SOLVE();
			double serr, ssum;
			ssum = 0.;
			for (i = 2; i <= NIM; i++) {
				serr = TSSOR[i] + AW[i] * TS[i - 1] + AE[i] * TS[i + 1] - AP[i] * TS[i];
				ssum = ssum + pow(serr, 2.);
			}
			serr = sqrt(ssum / (NI - 2));
			printf(" iter = %2d, serr = %15.7e \n", iter, serr);

			//
			if ((gerr < ERRMAX) && (serr < ERRMAX)) break;
		}
		
		// 다음 시간 단계 진행을 위해 현재 해를 이전 해 배열에 저장
		for (i = 1; i <= NI; i++) {
			TG_old[i] = TG[i];
			TS_old[i] = TS[i];
		}

		// 최종 결과 파일 저장
		if (fabs(fmod(t, 0.1)) < 1e-6) {
			char filename_g[50], filename_s[50];
			sprintf_s(filename_g, "outg_%.1f.txt", (float) t);
			sprintf_s(filename_s, "outs_%.1f.txt", (float) t);

			FILE* fp1 = _fsopen(filename_g, "w", _SH_DENYNO);
			for (i = 1; i <= NI; i++) {
				fprintf(fp1, " %4d %15.7e %15.7e \n", i, XP[i], TG[i]);
			}
			fclose(fp1);

			FILE* fp2 = _fsopen(filename_s, "w", _SH_DENYNO);
			for (i = 1; i <= NI; i++) {
				fprintf(fp2, " %4d %15.7e %15.7e \n", i, XP[i], TS[i]);
			}
			fclose(fp2);
			printf("Saved data at t = %f s\n", t);
		}
	}

	

	/*printf("Final Gas temperature \n");
	FILE* fp1 = _fsopen("outg", "w", _SH_DENYNO);
	for (i = 1; i <= NI; i++) {
		printf(" X, TG (%2d) = %15.7e  %15.7e \n", i, XP[i], TG[i]);
		fprintf(fp1, " %4d %15.7e %15.7e  \n", i, XP[i], TG[i]);
	}
	fclose(fp1);

	printf("Final Solid temperature \n");
	FILE* fp2 = _fsopen("outs", "w", _SH_DENYNO);
	for (i = 1; i <= NI; i++) {
		printf(" X, TS (%2d) = %15.7e  %15.7e \n", i, XP[i], TS[i]);
		fprintf(fp2, " %4d %15.7e %15.7e  \n", i, XP[i], TS[i]);
	}
	fclose(fp2);
	*/
	return 0;
}

void TG_SOLVE()
{
	for (i = 2; i <= NIM; i++) {
		AW[i] = KG / DXU[i] + MDOT * CPG / DXU[i];
		AE[i] = KG / DXU[i + 1];
		//--------Neumann
		if (i == NIM) AE[i] = 0.;
		//--------
		AP[i] = AW[i] + AE[i] + Utot / DXP[i] + (rhog * DXP[i] * CPG / dt); // (rhog * DXP[i] * CPG / dt 항 추가
		if (iter > 0) Tsolid = BTS[i]; else Tsolid = TS[i];
		TGSOR[i] = Utot / DXP[i] * Tsolid + (rhog * DXP[i] * CPG / dt) * TG_old[i]; //(rhog * DXP[i] * CPG / dt) * TG_old[i] 추가 + Utot / DXP[i]fh qusrud
		SS[i] = TGSOR[i];
	}
	// BC column
	SS[2] += AW[2] * TG[1];
	SS[NIM] += AE[NIM] * TG[NI];

	TDMA(2, NIM, TG);

	//Neumann
	TG[NI] = TG[NIM];
}

void TS_SOLVE()
{
	for (i = 2; i <= NIM; i++) {
		AW[i] = KS / DXU[i];
		AE[i] = KS / DXU[i + 1];
		//------Neumann
		if (i == 2) AW[i] = 0.;
		if (i == NIM) AE[i] = 0.;
		//-------
		if (iter > 0) Tgas = BTG[i]; else Tgas = TG[i];
		AP[i] = AW[i] + AE[i] + Utot / DXP[i] + (rhos * DXP[i] * CPG / dt); // (rhos * CPG / dt 항 추가
		TSSOR[i] = Utot / DXP[i] * Tgas + (rhos * DXP[i] * CPG / dt) * TS_old[i]; //(rhos * DXP[i] * CPG / dt) * TG_old[i] 추가
		SS[i] = TSSOR[i];
	}
	// BC column
	SS[2] += AW[2] * TS[1];
	SS[NIM] += AE[NIM] * TS[NI];

	TDMA(2, NIM, TS);
	// Neumann
	TS[1] = TS[2];
	TS[NI] = TS[NIM];
}

double TKK_INTP(double a, double b) { return 0.5 * (a + b); }

void INIT()
{
	NIM = NI - 1;

	for (i = 2; i <= NI; i++) XU[i] = DX0 * (i - 2.);
	for (i = 2; i <= NIM; i++) XP[i] = 0.5 * (XU[i + 1] + XU[i]);
	XP[1] = XU[2]; XP[NI] = XU[NI];
	for (i = 2; i <= NIM; i++) DXP[i] = XU[i + 1] - XU[i];
	for (i = 2; i <= NI; i++) DXU[i] = XP[i] - XP[i - 1];

	for (i = 2; i <= NIM; i++) { TG[i] = TGIN0; TS[i] = TSIN0; }
}

void TDMA(int ist, int iend, double x[IDIM])
{
	double beta, gama[IDIM];
	i = ist; beta = AP[i]; x[i] = SS[i] / beta;

	for (i = ist + 1; i <= iend; i++) {
		gama[i] = -AE[i - 1] / beta; beta = AP[i] + AW[i] * gama[i]; x[i] = (SS[i] + AW[i] * x[i - 1]) / beta;
	}

	for (i = iend - 1; i >= ist; i--) x[i] = x[i] - gama[i + 1] * x[i + 1];
}