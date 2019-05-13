#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
//#include <gsl/gsl_min.h>
#define MATHLIB_STANDALONE 1
#include <Rmath.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <float.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <iomanip>


//#include <R.h>
//#include "statsLib.h"
//#include "nr3.h"
//#include "ran.h"

using namespace std;

typedef vector<int> Row_Int;
typedef vector<Row_Int> Matrix_Int;

typedef vector<double> Row_Double;
typedef vector<Row_Double> Matrix_Double;
typedef vector<Matrix_Double> vector_3D;

void read_data(ofstream& A_out, ofstream& A_wolb_out, ofstream& lambda_out, ifstream& rel_siz_in, ifstream& rel_mod_in, ifstream& patch_probs_in);
double Lhood_dt(gsl_rng *r, ofstream& lambda_out, ofstream& A_out, ofstream& A_wolb_out, ofstream& release_flag_out, Matrix_Int neighbrs, Matrix_Int neighbrs_ne, vector<int> no_neighbrs, vector<int> no_neighbrs_ne);
void get_neighbours(Matrix_Int& loc_pos, Matrix_Int& neighbrs, Matrix_Int& neighbrs_ne, vector<int>& no_neighbrs, vector<int>& no_neighbrs_ne);
void init_neighbours(Matrix_Int& neighbrs, Matrix_Int& neighbrs_ne, vector<int>& no_neighbrs, vector<int>& no_neighbrs_ne);
void get_release_locns(Matrix_Int& loc_ind, vector<int>& release_flag);
void dispersal_new(Matrix_Int neighbrs, Matrix_Int neighbrs_ne, vector<int> no_neighbrs, vector<int> no_neighbrs_ne, Matrix_Double& A_vec, vector<double>& A_imm_vec, int age1, vector<int> lambda_class, int ta_steps);
void dispersal_new_simp(Matrix_Int neighbrs, Matrix_Int neighbrs_ne, vector<int> no_neighbrs, vector<int> no_neighbrs_ne, Matrix_Double& A_vec, int age1);
void dispersal_new_simp2(Matrix_Int neighbrs, Matrix_Int neighbrs_ne, vector<int> no_neighbrs, vector<int> no_neighbrs_ne, vector<double>& A_vec);

void write_loc_indices(ofstream& loc_ind_out, Matrix_Int& loc_ind);
void shift_vec3D(vector_3D the_vec, int the_size1, int the_size2, int the_size3);

const int no_cohorts = 28;
const int no_cohorts_real = 1300;
const int no_pdays = 170;
const int no_sdays = 169;
const int no_updates =0;
const int maxtime = 1100;
const int no_locns = 10201;
const int no_locns_x = 101;
const int no_locns_y = 101;
const int no_classes=8;
const int no_ages=6;

const int t_steps = 65;
const int c_steps = 40;


vector<float> surv(no_sdays), max_dt(no_cohorts), lambda_vec(8);

vector<double> AM(no_locns), AM_wolb(no_locns), A_ovipos_imm(t_steps*no_locns*no_classes),A_ovipos_wolb_imm(t_steps*no_locns*no_classes), AF_imm(no_ages*no_locns*no_classes), AF_wolb_imm(no_ages*no_locns*no_classes), L(t_steps*no_locns), L_wolb(t_steps*no_locns);

Matrix_Double mean_dt(no_cohorts, Row_Double(no_locns)), mean_dt_wolb(no_cohorts, Row_Double(no_locns)), std_dt(no_cohorts,Row_Double(no_locns)), std_dt_wolb(no_cohorts,Row_Double(no_locns)), mu_p(maxtime, Row_Double(no_locns)), mu_p_wolb(maxtime,Row_Double(no_locns)), AF(no_ages,Row_Double(no_locns)), A_ovipos(t_steps,Row_Double(no_locns)),  A_ovipos_wolb_rel(t_steps, Row_Double(no_locns)),AF_wolb(no_ages,Row_Double(no_locns)), A_ovipos_wolb(t_steps,Row_Double(no_locns)), AF_wolb_rel(no_ages, Row_Double(no_locns));

Matrix_Int loc_ind(no_locns, Row_Int(2));

vector<int> hdate(no_cohorts_real), loc_class(no_locns), release_flag(no_locns);
stringstream strstm;

double fit_cost, release_size, Plow, Phigh, rel_mod, move_prob, survA, surv_L;

int main(){

Matrix_Int neighbrs(no_locns, Row_Int(8)), neighbrs_ne(no_locns, Row_Int(8)), loc_pos(no_locns_x, Row_Int(no_locns_y));
vector<int> no_neighbrs(no_locns), no_neighbrs_ne(no_locns);

//Compiling version also in /JCU_c++_code/R-3.1.0/src/nmath/standalone

//GSL random number initialization
  const gsl_rng_type * T;
  gsl_rng * r;

  gsl_rng_env_setup();

  r = gsl_rng_alloc (gsl_rng_ranlxd1);
  gsl_rng_set(r,3);
 
//Define lambda_vec

lambda_vec.at(0) = 14; lambda_vec.at(1) = 11; lambda_vec.at(2) = 9; lambda_vec.at(3) = 7; lambda_vec.at(4) = 5; lambda_vec.at(5) = 3; lambda_vec.at(6) = 1.0; lambda_vec.at(7) = 0.5;  


//Input and Output files

ofstream L_avg_out, A_out, A_wolb_out, lambda_out, loc_ind_out, loc_class_out, no_nbrs_out, no_nbrs_ne_out, nbrs_out, nbrs_ne_out, loc_pos_out, release_flag_out;

ifstream rel_siz_in, rel_mod_in, patch_probs_in, sens_params_in;

//Read in the varying demographic parameters

  string sens_params_file;
  cin >> sens_params_file;
  strstm.clear();
  strstm.str("");
  strstm <<sens_params_file;
  string filename30=strstm.str();
  sens_params_in.open(filename30.c_str());

  sens_params_in >> survA;
  sens_params_in >> surv_L;
  sens_params_in >> move_prob;

  cout << "survA " << survA << " surv_L " << surv_L << " move_prob " << move_prob << endl;  

  sens_params_in.close();


read_data(A_out, A_wolb_out, lambda_out, rel_siz_in, rel_mod_in, patch_probs_in);

//make the productivity landscape
loc_class_out.open("loc_class_file.txt");
double ran;

for (int x=0; x<no_locns; x++) {
	ran = gsl_ran_flat(r,0,1);//ranflat[R1_int]; ++R1_int;//gsl_ran_flat(r,0,1);	
	if (ran>=Phigh) loc_class[x]=2;
	if (ran<Phigh && ran>=Plow) loc_class[x]=1;
	if (ran<Plow) loc_class[x]=0;
	//cout << " ran " << ran <<" x " << x << " loc_class " << loc_class[x] << " Phigh " << Phigh << " Plow " << Plow <<endl;
	//loc_class_out << loc_class[x] << endl;
}


loc_ind_out.open("loc_ind_file.txt");
write_loc_indices(loc_ind_out, loc_ind);

no_nbrs_out.open("no_nbrs.txt");
no_nbrs_ne_out.open("no_nbrs_ne.txt");
nbrs_out.open("nbrs.txt");
nbrs_ne_out.open("nbrs_ne.txt");
loc_pos_out.open("loc_pos.txt");
release_flag_out.open("release_flag.txt");

//initialize neighbour arrays FIRST
init_neighbours(neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne);

get_neighbours(loc_pos, neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne);

for (int i=0; i<no_locns; i++){
        no_nbrs_out << no_neighbrs[i] << endl;
        no_nbrs_ne_out << no_neighbrs_ne[i] << endl;
	//release_flag_out << release_flag[i] << endl;
        for (int j=0; j<8; j++) {
                nbrs_out << neighbrs[i][j] << " ";
                nbrs_ne_out << neighbrs_ne[i][j] << " ";
        }
        nbrs_out << endl;
        nbrs_ne_out << endl;
}
for (int y=no_locns_y-1; y>=0; y--){
        for (int x=0; x<no_locns_x; x++){
                loc_pos_out << loc_pos[x][y] << " ";
        }
        loc_pos_out << endl;
}

//Assign Cohorts to blocks and maximum development time
vector<int> max_dt(no_cohorts);
for (int i=0; i<no_cohorts; i++){
	//max_dt.at(i) = pdate.at(no_pdays-1) - hdate.at(i);
	max_dt.at(i) = maxtime - hdate.at(i);
	//cout <<"i " << i << " max_dt " << max_dt.at(i) << endl;
}

std::cout <<std::setprecision(10) << Lhood_dt(r, lambda_out, A_out, A_wolb_out,release_flag_out, neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne) << endl;

return 0;

}

//---------------------------------function "read_data"---------------------------------------
void read_data(ofstream& A_out, ofstream& A_wolb_out, ofstream& lambda_out, ifstream& rel_siz_in, ifstream& rel_mod_in, ifstream& patch_probs_in){

ifstream hdate_data, hatch_data;
hdate_data.open("hdate3.txt");
for (int i=0; i<no_cohorts_real; i++){
	hdate_data >> hdate.at(i);
}

//Open output files named in inits file
string A_file, A_wolb_file, rel_siz_file, rel_mod_file, patch_probs_file, lambda_file;
cin >> A_file >> A_wolb_file >> lambda_file >> rel_siz_file >> rel_mod_file >> patch_probs_file;
cout << " A_file "<<  A_file  << " A_wolb_file " << A_wolb_file <<  " release_siz_file " << rel_siz_file << " rel mod file " << rel_mod_file << " patch_probs_file " << patch_probs_file << " lambda_file " << lambda_file << endl;

strstm.clear();
strstm.str("");
strstm <<A_file;
string filename17=strstm.str();
A_out.open(filename17.c_str());

strstm.clear();
strstm.str("");
strstm <<A_wolb_file;
string filename20=strstm.str();
A_wolb_out.open(filename20.c_str());

strstm.clear();
strstm.str("");
strstm <<rel_siz_file;
string filename24=strstm.str();
rel_siz_in.open(filename24.c_str());

strstm.clear();
strstm.str("");
strstm <<lambda_file;
string filename26=strstm.str();
lambda_out.open(filename26.c_str());

strstm.clear();
strstm.str("");
strstm <<patch_probs_file;
string filename28=strstm.str();
patch_probs_in.open(filename28.c_str());

strstm.clear();
strstm.str("");
strstm <<rel_mod_file;
string filename29=strstm.str();
rel_mod_in.open(filename29.c_str());

rel_siz_in >> release_size;
cin >> fit_cost;
rel_mod_in >> rel_mod;
patch_probs_in >> Phigh >> Plow;

cout << " release_size " << release_size <<  " fit_cost " << fit_cost << " rel_mod " << rel_mod << endl;


}

//---------------------------------function "Lhood_dt"----------------------------------------
double Lhood_dt(gsl_rng *r, ofstream& lambda_out, ofstream& A_out, ofstream& A_wolb_out, ofstream& release_flag_out, Matrix_Int neighbrs, Matrix_Int neighbrs_ne, vector<int> no_neighbrs, vector<int> no_neighbrs_ne){

Matrix_Double non_emerg_prob(no_cohorts, Row_Double(no_locns)), non_emerg_prob_wolb(no_cohorts, Row_Double(no_locns)),  mn_gam(no_cohorts, Row_Double(no_locns)), mn_wolb_gam(no_cohorts, Row_Double(no_locns)),st_gam(no_cohorts, Row_Double(no_locns)),st_wolb_gam(no_cohorts, Row_Double(no_locns)), L_avg_cohort(no_cohorts, Row_Double(no_locns)), no_emerg_tot(no_cohorts, Row_Double(no_locns)), no_emerg_tot_wolb(no_cohorts, Row_Double(no_locns));

Matrix_Int emerg_flag(no_cohorts, Row_Int(no_locns));

//Make an array to store the number of larvae of each age at each time step
//Make an array to store the number of individuals from each cohort that emerge as pupae at each time 
vector<double> emerg_record((t_steps+1)*no_cohorts*no_locns),emerg_record_wolb((t_steps+1)*no_cohorts*no_locns), L_cohort(2*no_cohorts*no_locns),L_cohort_wolb(2*no_cohorts*no_locns);

Matrix_Double P(3, Row_Double(no_locns)), P_wolb(3, Row_Double(no_locns));
vector<int> lambda_class(no_locns), cohort_hdate(no_cohorts), cohort_round_vec(no_cohorts);

vector<double> freqA(no_locns);

//Parameters
double L_avg, L_avg2, L_cum2, L_cum, denom2, Dt_shp, Dt_shp_wolb, Dt_scl, Dt_scl_wolb, prob, prob_wolb, no_emerge,no_emerge_wolb, av_growth, L_av1, L_av2, L_avg_f, sh, w, lambda_max, survA_rel, intc_TL, intc_TL_wolb, alpha_TL, alpha_TL_wolb, exp_TL, exp_TL_wolb, a_value, a_value_wolb, b_value, b_value2, b_value2_wolb, intc_f, alpha_f, survA_wolb, lambda1, freq_end, lambda_min, mean_max,std_max, freq_end2, freqL_end, hatch_sim, hatch_sim_wolb, A_tot, A_wolb_tot, intc_TL_hf, intc_TL_lf, alpha_TL_hf, alpha_TL_lf, exp_TL_hf, exp_TL_lf, a_value_hf, a_value_lf, b_value_hf, b_value_lf, b_value2_hf, b_value2_lf,class_ran, ran, AM1,AM_wolb1,AF1,AF_wolb1,AF_wolb_rel1,A_ovipos1,A_ovipos_wolb1,A_ovipos_wolb_rel1,A_ovipos_imm1,A_ovipos_wolb_imm1,AF_imm1,AF_wolb_imm1, AM2,AM_wolb2,AF2,AF_wolb2,AF_wolb_rel2,A_ovipos2,A_ovipos_wolb2,A_ovipos_wolb_rel2,A_ovipos_imm2,A_ovipos_wolb_imm2,AF_imm2,AF_wolb_imm2,A_tot_in_rel,A_tot_wolb_in_rel;

int index,first_hatch_date, max_cohort, time_lag, count, tlagh, tlagg,tlagi,tlagp,tlagm, min_cohort, release_end, release_day_init, dens_lag, freq_end_day, release_locn, cohort_round,cohort_out_flag,release_day;

int cohort_posn, cohort_real;


first_hatch_date=hdate.at(0);
tlagh = 5;//approx lag between oviposition and hatching
tlagg = 6;
tlagi = 21;//time over which larval density is averaged
tlagp = 2;
tlagm = 2;


intc_TL_hf = 1.2;
intc_TL_lf = 19.6;
alpha_TL_hf = 0.154;
alpha_TL_lf = 2.74;
exp_TL_hf = 0.68;
exp_TL_lf = 0.43;
a_value_hf = 12.0;
a_value_lf = -0.97;
b_value_hf = 0.24;
b_value_lf = 1.0;
b_value2_hf = 0.44;
b_value2_lf = 0.32;


//Wolbachia parameters
sh=0.99;
w=0.01;

release_locn = 596;

lambda_max=14;
lambda_min=0.5;
mean_max = 60;
std_max = 40;

release_day=400;
release_day_init = release_day;
release_end = release_day + 105;
freq_end_day = release_end + 180;

survA_rel=survA-fit_cost;
survA_wolb = survA-fit_cost;

ofstream L_cohort_out;
L_cohort_out.open("L_cohort_file.txt");

//Initialise L_cohort
for (int i=0; i<=1; i++){
	for (int j=0; j<no_cohorts ; j++){
		for (int x=0; x<no_locns; x++){
		L_cohort[i + 2*j + 2*no_cohorts*x]=0;
		L_cohort_wolb[i + 2*j + 2*no_cohorts*x]=0;
		}
	}
}


//Initialise L_update
for (int i=0; i<t_steps; i++) {
	for (int x=0; x<no_locns; x++){
		L[i+t_steps*x]=0; L_wolb[i+t_steps*x]=0;
	}
}

//Initialise emerg_record array.  
for (int i=0; i<=t_steps; i++){
	for (int j=0; j<no_cohorts; j++){
		for (int x=0; x<no_locns; x++){
		emerg_record[i + (t_steps+1)*j + (t_steps+1)*no_cohorts*x]=0;
		emerg_record_wolb[i + (t_steps+1)*j + (t_steps+1)*no_cohorts*x]=0;
		}
	}
}

//Initialise non_emerg_prob array.  
for (int i=0; i<no_cohorts; i++){
	for (int x=0; x<no_locns; x++){
	non_emerg_prob[i][x]=1;
	non_emerg_prob_wolb[i][x]=1;
	}
}

//Initialise emerg_flag
for (int i=0; i<no_cohorts; i++) {
	for (int x=0; x<no_locns; x++){
		emerg_flag[i][x]=0;
	}
}

//Initialise L_avg_cohort & L_avg
for (int i=0; i<no_cohorts; i++) {
	for (int x=0; x<no_locns; x++){
	L_avg_cohort[i][x] = 0;
	}
}

//Clear mu_p_update
for (int i=0; i<maxtime; i++){
	for (int x=0; x<no_locns; x++){
	mu_p[i][x] = 0; 
	mu_p_wolb[i][x]=0;
	}
}

//Clear mn_update
for (int i=0; i<no_cohorts; i++) {
	for (int x=0; x<no_locns; x++){
		mn_gam[i][x] = 999; mn_wolb_gam[i][x] = 999;
	}
}


//Clear st_update
for (int i=0; i<no_cohorts; i++) {
	for (int x=0; x<no_locns; x++){
		st_gam[i][x] = 1; st_wolb_gam[i][x]=1;
	}
}

for (int i=0; i<no_cohorts; i++) {
	for (int x=0; x<no_locns; x++){
	mean_dt[i][x]=0;
	mean_dt_wolb[i][x]=0;
	std_dt[i][x]=0;
	std_dt_wolb[i][x]=0;
	no_emerg_tot[i][x]=0;
	no_emerg_tot_wolb[i][x]=0;
	}
}


//Initialize P and P_wolb
for (int i=0; i<3; i++){
	for (int x=0; x<no_locns; x++){
		P[i][x]=0;
		P_wolb[i][x]=0;
	}
}

//Initialize AF
for (int x=0; x<no_locns; x++) AF[0][x] = 1;
for (int i=1; i<no_ages; i++) {
	for (int x=0; x<no_locns; x++){
		AF[i][x]=0;
	}
}

//Initialize AF_wolb
for (int i=0; i<no_ages; i++){
	for (int x=0; x<no_locns; x++){	
 		AF_wolb[i][x]=0;
		AF_wolb_rel[i][x] = 0;
	}
}

//Initialize AF_imm and AF_wolb_imm
for (int i=0; i<no_ages; i++){
	for (int x=0; x<no_locns; x++){
		for (int c=0; c<no_classes; c++){
			AF_imm[i+no_ages*x+no_ages*no_locns*c]=0;
			AF_wolb_imm[i+no_ages*x+no_ages*no_locns*c]=0;

		}
	}
}
 
//Initialize A_ovipos and A_ovipos_wolb
for (int i=0; i<t_steps; i++) {
	for (int x=0; x<no_locns; x++)	{
				A_ovipos[i][x] = 0;
				A_ovipos_wolb[i][x] = 0;
				A_ovipos_wolb_rel[i][x] = 0;
				}
}

//Initialize A_ovipos_imm and A_ovipos_wolb_imm
for (int i=0; i<t_steps; i++) {
	for (int x=0; x<no_locns; x++){
		for (int c=0; c<no_classes; c++){
			A_ovipos_imm[i + t_steps*x + t_steps*no_locns*c]=0;
			A_ovipos_wolb_imm[i + t_steps*x + t_steps*no_locns*c]=0;
		}
	}
}

//Initialize AM
for (int x=0; x<no_locns; x++) {
	AM[x]=1;
	AM_wolb[x]=0;
}

//Initialize freqA
for (int x=0; x<no_locns; x++) freqA[x]=0;

//Initialize cohort_hdate
for (int cohort=0; cohort < no_cohorts; cohort++) cohort_hdate[cohort]=hdate.at(cohort);

A_tot_in_rel=0;
A_tot_wolb_in_rel=0;

cohort_real=-1;
cohort_posn=-1;
cohort_round=0;
for (int time1=1; time1<maxtime; time1++){
	cohort_out_flag=0;
	if (time1 == hdate.at(cohort_real+1)) {++cohort_posn; ++cohort_real;}
	cout << "time1 " << time1 <<" cohort " << cohort_real <<endl;

	if (time1==release_day_init){
		for (int x=0; x<no_locns; x++) release_flag[x]=0;
		get_release_locns(loc_ind,release_flag);
		for (int x=0; x<no_locns; x++) release_flag_out << release_flag[x] << endl;
	}

	for (int locn=0; locn<no_locns; locn++){
		//cout << "locn " << locn << endl;

		//update pupae age classes
		P[2][locn]=P[1][locn]*survA;
		P[1][locn]=P[0][locn]*survA;
		P[0][locn]=0;
		P_wolb[2][locn]=P_wolb[1][locn]*survA_wolb;
		P_wolb[1][locn]=P_wolb[0][locn]*survA_wolb;
		P_wolb[0][locn]=0;


		//accomodating for fast oviposition on first cohort
		if (time1<=hdate.at(2)) tlagg=5;
		else tlagg=6;
	
		for (int cohort=0; cohort<no_cohorts; cohort++){
		//	cout << "cohort " << cohort << endl;
	
			//kill larvae in empty houses
			if (loc_class[locn]==0) {L_cohort[1 + 2*cohort + 2*no_cohorts*locn]=0;L_cohort_wolb[1 + 2*cohort + 2*no_cohorts*locn]=0;
				L_cohort[0 + 2*cohort + 2*no_cohorts*locn]=0;L_cohort_wolb[0 + 2*cohort + 2*no_cohorts*locn]=0;
			}
			//update larvae age classes
			L_cohort[1 + 2*cohort + 2*no_cohorts*locn] = L_cohort[0 + 2*cohort + 2*no_cohorts*locn] *  surv_L;
			L_cohort_wolb[1+ 2*cohort + 2*no_cohorts*locn] = L_cohort_wolb[0 + 2*cohort + 2*no_cohorts*locn] * surv_L;
		}

		if (cohort_real>=0 && time1 == hdate.at(cohort_real)){

			//One week larval density lagged by 6 days
			L_avg = 0;
			dens_lag = time1-tlagh-tlagi-tlagg;
			if (dens_lag>=hdate.at(0)){
			//lags: between oviposition and hatching, time over which larval density is averaged, time to become gravid after pupation
				for (int j=tlagh+tlagi+tlagg; j>tlagh+tlagg; j--) L_avg+=L[j + t_steps*locn] + L_wolb[j+t_steps*locn];
					L_avg = L_avg/tlagi; 
				}
				else if (dens_lag>=0) {
					if (time1-tlagh-tlagg<=hdate.at(0)) L_avg=0;//if the end of the lag is before the first hatch date
					if (time1-tlagh-tlagg>hdate.at(0)) {
						for (int j=time1-hdate.at(0)-1; j>=tlagh+tlagg; j--) L_avg+=L[j+t_steps*locn] + L_wolb[j+t_steps*locn];
						L_avg = L_avg/tlagi;
					
					} 
				}
				else L_avg = 0; 

			//Work out the fecundity classes to apply to each cohort hatched from the local subpopulation
				//high quality
			if (loc_class[locn]==2){
				if (L_avg<=100) lambda_class[locn] = 0;
				if (L_avg>100 && L_avg<=250) lambda_class[locn] = 1;
				if (L_avg>250 && L_avg<=550) lambda_class[locn] = 2;
				if (L_avg>550 && L_avg<=1350) lambda_class[locn] = 3;
				if (L_avg>1350 && L_avg<=2140) lambda_class[locn] = 4;
				if (L_avg>2140 &&L_avg<=4300) lambda_class[locn] = 5;
				if (L_avg>4300 && L_avg<=5250) lambda_class[locn] = 6;
				if (L_avg>5250) lambda_class[locn] = 7;
			}
				//low quality
			if (loc_class[locn]==1){
				if (L_avg<=5) lambda_class[locn] = 0;

				if (L_avg>5 && L_avg<=10) lambda_class[locn] = 1;

				if (L_avg>10 && L_avg<=25) lambda_class[locn] = 2;

				if (L_avg>25 && L_avg<=50) lambda_class[locn] = 3;

				if (L_avg>50 && L_avg<=100) lambda_class[locn] = 4;
				if (L_avg>100 && L_avg<=660) lambda_class[locn] = 5;
				if (L_avg>660 && L_avg<=975) lambda_class[locn] = 6;
				if (L_avg>975) lambda_class[locn] = 7;
			}
			//Work out the numbers of larvae hatched at each location and cohort
 			if (time1>tlagh+tlagg && loc_class[locn]>0){
				lambda_out << lambda_vec.at(lambda_class[locn]) << " ";
				hatch_sim=0; hatch_sim_wolb=0;
				for (int classes=0; classes<no_classes; classes++){
					if (lambda_class[locn] == classes) {
						lambda1 = lambda_vec.at(classes);
						hatch_sim += lambda1*A_ovipos[tlagh][locn] + w*lambda1*A_ovipos_wolb[tlagh][locn]+ w*lambda_max*A_ovipos_wolb_rel[tlagh][locn];
						hatch_sim_wolb += lambda1*(1-w)*A_ovipos_wolb[tlagh][locn] + lambda_max*(1-w)*A_ovipos_wolb_rel[tlagh][locn];
					}
					hatch_sim+=lambda_vec[classes]*A_ovipos_imm[tlagh + t_steps*locn + t_steps*no_locns*classes] + w*lambda_vec[classes]*A_ovipos_wolb_imm[tlagh + t_steps*locn + t_steps*no_locns*classes];
					hatch_sim_wolb+=lambda_vec[classes]*(1-w)*A_ovipos_wolb_imm[tlagh + t_steps*locn + t_steps*no_locns*classes];

				}
			}
			if (loc_class[locn]==0) {hatch_sim=0;hatch_sim_wolb=0;}

			L_cohort[1+ 2*cohort_posn + 2*no_cohorts*locn] = hatch_sim;
			L_cohort_wolb[1+ 2*cohort_posn + 2*no_cohorts*locn] = hatch_sim_wolb;

			cohort_hdate[cohort_posn] = time1;
			cohort_round_vec[cohort_posn] = cohort_round;
			emerg_flag[cohort_posn][locn]=0;
			mn_gam[cohort_posn][locn]=0;
			mn_wolb_gam[cohort_posn][locn]=0;
			st_gam[cohort_posn][locn]=0;
			st_wolb_gam[cohort_posn][locn]=0;
			L_avg_cohort[cohort_posn][locn]=0;
			for (int i=0; i<=t_steps; i++) {emerg_record[i+(t_steps+1)*cohort_posn + (t_steps+1)*no_cohorts*locn]=0;emerg_record_wolb[i+(t_steps+1)*cohort_posn + (t_steps+1)*no_cohorts*locn]=0;}
			non_emerg_prob[cohort_posn][locn]=1;
			non_emerg_prob_wolb[cohort_posn][locn]=1;
			no_emerg_tot[cohort_posn][locn]=0;
			no_emerg_tot_wolb[cohort_posn][locn]=0;
			mean_dt[cohort_posn][locn]=0;
			mean_dt_wolb[cohort_posn][locn]=0;
			std_dt[cohort_posn][locn]=0;
			std_dt_wolb[cohort_posn][locn]=0;

		}//end of if time1==hdate.at(cohort) loop	
		
		for (int cohort=0; cohort<no_cohorts; cohort++){

			if (time1>cohort_hdate[cohort]+4 && emerg_flag[cohort][locn]==0){ //if the cohort could have begun emergence and less than 1 individual have emerged so far
		//check that this ordering of the cohorts and times is okay
 				L_avg2 = 0;
				L_cum2 = 0;
				denom2 = 0;
				for (int etime = max(time1-t_steps,cohort_hdate[cohort]+5); etime <=time1; etime++){//all emergence times "etime" past to present
					int e_index = time1-etime;
					L_cum=0;
					count=cohort;
					for (int k=min(time1 - cohort_hdate[cohort], t_steps-1); k>e_index; k--) {
						L_cum += (L[k+t_steps*locn] + L_wolb[k+t_steps*locn]);//cumulative exposure experienced at day before emergence time
					}
					L_cum2 += (emerg_record[e_index + (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn] + emerg_record_wolb[e_index + (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn])*L_cum/(etime-cohort_hdate[cohort]);//accumulating daily average exposure * no. emerged
					denom2 += emerg_record[e_index + (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn] + emerg_record_wolb[e_index + (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn];//cumulative no. emerged
				}
				if (denom2>0) L_avg2 = L_cum2/denom2;
				else L_avg2 = L_cum/(time1 - cohort_hdate[cohort]);

				if (loc_class[locn]==2){intc_TL = intc_TL_hf; alpha_TL = alpha_TL_hf; exp_TL = exp_TL_hf; a_value = a_value_hf; b_value = b_value_hf; b_value2 = b_value2_hf;}
				if (loc_class[locn]==1){intc_TL = intc_TL_lf; alpha_TL = alpha_TL_lf; exp_TL = exp_TL_lf; a_value = a_value_lf; b_value = b_value_lf; b_value2 = b_value2_lf;}

				mn_gam[cohort][locn] = intc_TL + alpha_TL * pow(L_avg2, exp_TL);
				if (L_avg2<50) mn_gam[cohort][locn]=15;
				mn_wolb_gam[cohort][locn] = mn_gam[cohort][locn];

				if (mn_gam[cohort][locn]<0) mn_gam[cohort][locn]=0;
				if (mn_wolb_gam[cohort][locn]<0) mn_wolb_gam[cohort][locn]=0;
				if (mn_gam[cohort][locn]>mean_max) mn_gam[cohort][locn] = mean_max;
				if (mn_wolb_gam[cohort][locn]>mean_max) mn_wolb_gam[cohort][locn] = mean_max;

				if (denom2>0) {
					emerg_flag[cohort][locn]=1;
					L_avg_cohort[cohort][locn] = L_avg2;
				}

				st_gam[cohort][locn] = a_value + b_value * pow(L_avg2, b_value2);
				if (L_avg2<100) st_gam[cohort][locn]=a_value + b_value*pow(100,b_value2);

				st_wolb_gam[cohort][locn] = st_gam[cohort][locn];

				if (st_gam[cohort][locn]>std_max) st_gam[cohort][locn] = std_max;
				if (st_wolb_gam[cohort][locn]>std_max) st_wolb_gam[cohort][locn] = std_max;

			
			}//end of if (time1>hdate.at(cohort)+4 && emerg_flag.at(cohort)==0)
			if (time1>cohort_hdate[cohort]+4){
			
				Dt_shp = pow(mn_gam[cohort][locn]/st_gam[cohort][locn],2);
				Dt_scl = mn_gam[cohort][locn]/Dt_shp;			
				Dt_shp_wolb = pow(mn_wolb_gam[cohort][locn]/st_wolb_gam[cohort][locn],2);
				Dt_scl_wolb = mn_wolb_gam[cohort][locn]/Dt_shp_wolb;

				if (mn_gam[cohort][locn]==1){
					if (loc_class[locn]==2){
						Dt_shp = 9.0;
						Dt_scl = 0.2;
						Dt_shp_wolb = 9.0;
						Dt_scl_wolb = 0.2;
					}
					if (loc_class[locn]==2){
						Dt_shp = 9.0;
						Dt_scl = 0.8;
						Dt_shp_wolb = 9.0;
						Dt_scl_wolb = 0.8;
					}
				}
			
				time_lag = time1 - cohort_hdate[cohort];
				if (time_lag>5 && mn_gam[cohort][locn]>0 && mn_gam[cohort][locn]<mean_max && Dt_shp>0 && Dt_scl>0 && Dt_shp<200 && Dt_scl<200){
					prob = gsl_cdf_gamma_P(time_lag-5,Dt_shp,Dt_scl) - gsl_cdf_gamma_P(time_lag-5-1,Dt_shp,Dt_scl);
					if (time1>hdate.at(no_cohorts_real-1)) prob*= sqrt((maxtime-hdate.at(no_cohorts_real-1))/(maxtime-time1+1));

				}
				else prob=0;
				if (time_lag>5 && mn_wolb_gam[cohort][locn]>0 && mn_wolb_gam[cohort][locn]<mean_max && Dt_shp_wolb>0 && Dt_scl_wolb>0 && Dt_shp_wolb<200 && Dt_scl_wolb<200){
					prob_wolb = gsl_cdf_gamma_P(time_lag-5,Dt_shp_wolb,Dt_scl_wolb) - gsl_cdf_gamma_P(time_lag-5-1,Dt_shp_wolb,Dt_scl_wolb);
					if (time1>hdate.at(no_cohorts_real-1)) prob_wolb*= sqrt((maxtime-hdate.at(no_cohorts_real-1))/(maxtime-time1+1));
				}
				else prob_wolb=0;

				if (mn_gam[cohort][locn]==0){	 
					Dt_shp = 9.0;
			   		Dt_scl = 0.2;
			   		prob = gsl_cdf_gamma_P(time_lag-4,Dt_shp,Dt_scl) - gsl_cdf_gamma_P(time_lag-4-1,Dt_shp,Dt_scl);
				}
				if (mn_wolb_gam[cohort][locn]==0){
			   		Dt_shp_wolb = 9.0;
			    		Dt_scl_wolb = 0.2;
			   		prob_wolb = gsl_cdf_gamma_P(time_lag-4,Dt_shp_wolb,Dt_scl_wolb) - gsl_cdf_gamma_P(time_lag-4-1,Dt_shp_wolb,Dt_scl_wolb);
				}

				no_emerge = L_cohort[0+ 2*cohort + 2*no_cohorts*locn] * prob/non_emerg_prob[cohort][locn];
				no_emerge_wolb = L_cohort_wolb[0+ 2*cohort + 2*no_cohorts*locn] * prob_wolb/non_emerg_prob_wolb[cohort][locn];


				if (no_emerge > L_cohort[0+ 2*cohort + 2*no_cohorts*locn]){
			 		no_emerge=L_cohort[0+ 2*cohort + 2*no_cohorts*locn];
			 		prob=1;
				}

				if (no_emerge<0.001){no_emerge=0; prob=0;}
				if (prob<0) prob=0;

				if (no_emerge_wolb > L_cohort_wolb[0+ 2*cohort + 2*no_cohorts*locn]){
					no_emerge_wolb=L_cohort_wolb[0+ 2*cohort + 2*no_cohorts*locn];
			 		prob_wolb=1;
				}

				if (no_emerge_wolb<0.001){no_emerge_wolb=0; prob_wolb=0;}
				if (prob_wolb<0) prob_wolb=0;

				non_emerg_prob[cohort][locn] *= (1-prob/non_emerg_prob[cohort][locn]);
				non_emerg_prob_wolb[cohort][locn] *= (1-prob_wolb/non_emerg_prob_wolb[cohort][locn]);

				L_cohort[1+ 2*cohort + 2*no_cohorts*locn] -= no_emerge;
				L_cohort_wolb[1+ 2*cohort + 2*no_cohorts*locn] -= no_emerge_wolb;
				emerg_record[0 + (t_steps+1)*cohort +(t_steps+1)*no_cohorts*locn]=no_emerge;
				emerg_record_wolb[0 + (t_steps+1)*cohort +(t_steps+1)*no_cohorts*locn]=no_emerge_wolb;
				mean_dt[cohort][locn]+=(time1-cohort_hdate[cohort])*no_emerge;
				mean_dt_wolb[cohort][locn]+=(time1-cohort_hdate[cohort])*no_emerge_wolb;
				no_emerg_tot[cohort][locn]+=no_emerge;
				no_emerg_tot_wolb[cohort][locn]+=no_emerge_wolb;
				mu_p[time1][locn]+=no_emerge;
				mu_p_wolb[time1][locn]+=no_emerge_wolb;

				P[0][locn] += no_emerge;
				P_wolb[0][locn] += no_emerge_wolb;
			}//end of if (time1>hdate.at(cohort)+4) loop

			L[0+t_steps*locn]+=L_cohort[1+ 2*cohort + 2*no_cohorts*locn];//needs to go at the end to take into account losses from emergence
			L_wolb[0+t_steps*locn]+=L_cohort_wolb[1+ 2*cohort + 2*no_cohorts*locn];


//work out cohort development time means and standard deviations

			if (time1-cohort_hdate[cohort]==t_steps-1){
				cohort_out_flag=1;
				if (no_emerg_tot[cohort][locn]>0) {mean_dt[cohort][locn]/=no_emerg_tot[cohort][locn];
				if (mean_dt[cohort][locn]>100) cout << "cohort " << cohort << " locn " << locn << " mean_dt " << mean_dt[cohort][locn] << " no_emerg_tot " << no_emerg_tot[cohort][locn] << endl;
				}

				else mean_dt[cohort][locn]=0;

				if (no_emerg_tot_wolb[cohort][locn]>0) mean_dt_wolb[cohort][locn]/=no_emerg_tot_wolb[cohort][locn];
				else mean_dt_wolb[cohort][locn]=0;

				if  (no_emerg_tot[cohort][locn]>0){
				for (int j=0; j<=t_steps; j++){
					std_dt[cohort][locn] += (time1-j-cohort_hdate[cohort]-mean_dt[cohort][locn])*(time1-j-cohort_hdate[cohort]-mean_dt[cohort][locn])*emerg_record[j + (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn];
					std_dt_wolb[cohort][locn] += (time1-j-cohort_hdate[cohort]-mean_dt_wolb[cohort][locn])*(time1-j-cohort_hdate[cohort]-mean_dt_wolb[cohort][locn])*emerg_record_wolb[time1-j-cohort_hdate[cohort] + (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn];
				}
				std_dt[cohort][locn] = sqrt(std_dt[cohort][locn]/no_emerg_tot[cohort][locn]);
				std_dt_wolb[cohort][locn] = sqrt(std_dt_wolb[cohort][locn]/no_emerg_tot_wolb[cohort][locn]);
				//cout << "time1 " << time1 << " cohort " << cohort << " locn " << locn <<" std_dt " << std_dt[cohort][locn] << endl;
				}
				else {std_dt[cohort][locn]=0; std_dt_wolb[cohort][locn]=0;}
			}

			//shift emerg_record
			for (int j=t_steps-1; j>=0; j--) {emerg_record[j+1 + (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn] = emerg_record[j+ (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn];
				emerg_record_wolb[j+1+ (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn] = emerg_record_wolb[j+ (t_steps+1)*cohort + (t_steps+1)*no_cohorts*locn];
			}
		}//end of cohort loop

		
//-------------ADULT SUBMODEL ----------------------------------
		A_tot=0;
		A_wolb_tot=0;

		//locals: aged less than minimum for oviposition	
		for (int i=no_ages-1; i>0; i--){
			AF[i][locn] = AF[i-1][locn]*survA;
			AF_wolb[i][locn] = AF_wolb[i-1][locn]*survA_wolb;
			A_tot += AF[i][locn];
			A_wolb_tot+=AF_wolb[i][locn];
		}
		AF[0][locn]=0;AF_wolb[0][locn]=0;

		//locals: old enough for oviposition
		A_ovipos[0][locn] = A_ovipos[1][locn] *survA; 
		A_ovipos_wolb[0][locn]  = A_ovipos_wolb[1][locn] *survA_wolb;

		A_tot += A_ovipos[0][locn];
		A_wolb_tot += A_ovipos_wolb[0][locn];

		A_ovipos[0][locn] += AF[no_ages-1][locn];
		A_ovipos_wolb[0][locn] += AF_wolb[no_ages-1][locn];

		AF[0][locn] = 0.5*(1-sh*freqA[locn])*P[2][locn];
		AF_wolb[0][locn] = 0.5*P_wolb[2][locn];

		A_tot += AF[0][locn];
		A_wolb_tot += AF_wolb[0][locn];

		//immigrants
		for (int classes=0; classes<no_classes; classes++) {
			for (int i=no_ages-1; i>0; i--){
				AF_imm[i+no_ages*locn+no_ages*no_locns*classes] = AF_imm[i-1+no_ages*locn+no_ages*no_locns*classes]*survA;
				AF_wolb_imm[i+no_ages*locn+no_ages*no_locns*classes] = AF_wolb_imm[i-1+no_ages*locn+no_ages*no_locns*classes]*survA_wolb;
				A_tot+=AF_imm[i+no_ages*locn+no_ages*no_locns*classes];
				A_wolb_tot+=AF_wolb_imm[i+no_ages*locn+no_ages*no_locns*classes];
			}
			AF_imm[0+no_ages*locn + no_ages*no_locns*classes]=0;
			AF_wolb_imm[0+no_ages*locn + no_ages*no_locns*classes]=0;
			A_ovipos_imm[0 + t_steps*locn + t_steps*no_locns*classes] = A_ovipos_imm[1+ t_steps*locn + t_steps*no_locns*classes]*survA;
			A_ovipos_wolb_imm[0 + t_steps*locn + t_steps*no_locns*classes] = A_ovipos_wolb_imm[1+ t_steps*locn + t_steps*no_locns*classes]*survA_wolb; 

			A_tot+=A_ovipos_imm[0+ t_steps*locn + t_steps*no_locns*classes];					    	     A_wolb_tot+=A_ovipos_wolb_imm[0+ t_steps*locn + t_steps*no_locns*classes];	

			A_ovipos_imm[0+ t_steps*locn + t_steps*no_locns*classes] += AF_imm[no_ages-1+no_ages*locn + no_ages*no_locns*classes];
			A_ovipos_wolb_imm[0+ t_steps*locn + t_steps*no_locns*classes] += AF_wolb_imm[no_ages-1+no_ages*locn + no_ages*no_locns*classes];
		}

		AM[locn]*=survA;
		AM_wolb[locn]*=survA_wolb;
	
		//account for wolbachia additions 
		if (time1>release_day_init) {
			for (int i=no_ages-1; i>0; i--) {
				AF_wolb_rel[i][locn] = AF_wolb_rel[i-1][locn]*survA_rel;
				A_wolb_tot+=AF_wolb_rel[i][locn];
			}
			AF_wolb_rel[0][locn] = 0;

			A_ovipos_wolb_rel[0][locn] = A_ovipos_wolb_rel[1][locn]*survA_rel;
			A_ovipos_wolb_rel[0][locn] += AF_wolb_rel[no_ages-1][locn];
		}
		if (time1==release_day && time1<release_end && release_flag[locn]==1){
			AF_wolb_rel[0][locn]=0.5*release_size*rel_mod;
			A_wolb_tot+=AF_wolb_rel[0][locn];
			AM_wolb[locn]+=0.5*release_size*rel_mod;
		}

		AM[locn] += 0.5*P[2][locn];
		AM_wolb[locn] += 0.5*P_wolb[2][locn];	
		A_tot+=AM[locn];
		A_wolb_tot+=AM_wolb[locn];

		if (AM_wolb[locn]>0) freqA[locn] = AM_wolb[locn]/(AM[locn]+AM_wolb[locn]);
		else freqA[locn]=0;	
	
		//write total adult abundances 
		A_out << AM[locn] << " ";
		A_wolb_out << AM_wolb[locn] << " ";

		//shift L_cohort
		for (int i=0; i<no_cohorts; i++) {
			L_cohort[0+ 2*i + 2*no_cohorts*locn] =  L_cohort[1+ 2*i + 2*no_cohorts*locn];
			L_cohort_wolb[0+ 2*i + 2*no_cohorts*locn] = L_cohort_wolb[1+ 2*i + 2*no_cohorts*locn];
		}

		if (time1==release_end && release_flag[locn]==1){
			A_tot_in_rel+= AM[locn] + AM_wolb[locn];
			A_tot_wolb_in_rel+=AM_wolb[locn];
		}

	}//end of locn loop

	if (time1==release_end) {
		freq_end = A_tot_wolb_in_rel/A_tot_in_rel;
		cout << "freq_end " << A_tot_wolb_in_rel/A_tot_in_rel << " A_tot_in_rel " << A_tot_in_rel << " A_tot_wolb_in_rel " << A_tot_wolb_in_rel;
	}

	if (time1==release_day && time1<release_end) release_day+=7;

	//write average larval densities for eahc cohort and location
	if (cohort_posn==no_cohorts-1) {
		cohort_posn=-1;
		++cohort_round;
		//for (int i=0; i<no_cohorts; i++) {
			//for (int x=0; x<no_locns; x++){
				//L_avg_out << L_avg_cohort[i][x] << " ";
			//}
			//L_avg_out << endl;
		//}
		//L_avg_out << endl;
	}

	A_out << endl;
	A_wolb_out << endl;
	lambda_out << endl;



//test totals
/*AM1 =0;AM_wolb1=0;
AF1=0;AF_wolb1=0;AF_wolb_rel1=0;
A_ovipos1=0;A_ovipos_wolb1=0;A_ovipos_wolb_rel1=0;
A_ovipos_imm1=0;A_ovipos_wolb_imm1=0;AF_imm1=0;AF_wolb_imm1=0;
for (int x=0; x<no_locns; x++){
	AM1 += AM[x]; AM_wolb1 += AM_wolb[x];
	for (int ages=0; ages<no_ages; ages++) {
		AF1+=AF[ages][x]; AF_wolb1+=AF_wolb[ages][x];AF_wolb_rel1+=AF_wolb_rel[ages][x];
		for (int classes=0; classes<no_classes; classes++){
			AF_imm1+=AF_imm[ages+no_ages*x+no_ages*no_locns*classes];
			AF_wolb_imm1+=AF_wolb_imm[ages+no_ages*x+no_ages*no_locns*classes];
		}
	}
	for (int t=0; t<t_steps; t++){
		A_ovipos1 += A_ovipos[t][x]; A_ovipos_wolb1+=A_ovipos_wolb[t][x]; A_ovipos_wolb_rel1+=A_ovipos_wolb_rel[t][x];
		for (int classes=0; classes<no_classes; classes++){
			A_ovipos_imm1 += A_ovipos_imm[t+t_steps*x+t_steps*no_locns*classes];
			A_ovipos_wolb_imm1 += A_ovipos_wolb_imm[t+t_steps*x+t_steps*no_locns*classes];
		}
	}
}
*/

	//disperse local and immigrants of the early age classes

	for (int i=0; i<no_ages; i++) dispersal_new(neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne, AF, AF_imm, i, lambda_class, no_ages);
	if (time1>release_day_init) for (int i=0; i<no_ages; i++) dispersal_new(neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne, AF_wolb, AF_wolb_imm, i, lambda_class,no_ages);

	dispersal_new(neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne, A_ovipos, A_ovipos_imm, 0, lambda_class,t_steps);
	if (time1>release_day_init) {
		dispersal_new(neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne, A_ovipos_wolb, A_ovipos_wolb_imm, 0, lambda_class, t_steps);
		for (int i=0; i<no_ages; i++) dispersal_new_simp(neighbrs,neighbrs_ne, no_neighbrs, no_neighbrs_ne, AF_wolb_rel, i);
		dispersal_new_simp(neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne, A_ovipos_wolb_rel,0); 
	}

	dispersal_new_simp2(neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne, AM);
	if (time1>release_day_init) dispersal_new_simp2(neighbrs, neighbrs_ne, no_neighbrs, no_neighbrs_ne, AM_wolb);

	//test totals
	/*
	AM2 =0;AM_wolb2=0;
	AF2=0;AF_wolb2=0;AF_wolb_rel2=0;
	A_ovipos2=0;A_ovipos_wolb2=0;A_ovipos_wolb_rel2=0;
	A_ovipos_imm2=0;A_ovipos_wolb_imm2=0;AF_imm2=0;AF_wolb_imm2=0;
	for (int x=0; x<no_locns; x++){
		AM2 += AM[x]; AM_wolb2 += AM_wolb[x];
		for (int ages=0; ages<no_ages; ages++) {
			AF2+=AF[ages][x]; AF_wolb2+=AF_wolb[ages][x];AF_wolb_rel2+=AF_wolb_rel[ages][x];
			for (int classes=0; classes<no_classes; classes++){
				AF_imm2+=AF_imm[ages+no_ages*x+no_ages*no_locns*classes];
				AF_wolb_imm2+=AF_wolb_imm[ages+no_ages*x+no_ages*no_locns*classes];
			}
		}
		for (int t=0; t<t_steps; t++){
			A_ovipos2 += A_ovipos[t][x]; A_ovipos_wolb2+=A_ovipos_wolb[t][x]; A_ovipos_wolb_rel2+=A_ovipos_wolb_rel[t][x];
			for (int classes=0; classes<no_classes; classes++){
				A_ovipos_imm2 += A_ovipos_imm[t+t_steps*x+t_steps*no_locns*classes];
				A_ovipos_wolb_imm2 += A_ovipos_wolb_imm[t+t_steps*x+t_steps*no_locns*classes];
			}
		}
	}

	cout <<" AM " << AM1 << " " << AM2 << endl; cout << " AM_wolb " << AM_wolb1 << " " << AM_wolb2 << endl; cout << " AF " << AF1 + AF_imm1 << " " << AF2 + AF_imm2<< endl; cout << " AF_loc " << AF1 << " " << AF2 << endl; cout << " AF_wolb " << AF_wolb1 + AF_wolb_imm1 << " " << AF_wolb2 + AF_wolb_imm2 << endl;cout << " AF_wolb_rel " << AF_wolb_rel1 << " " << AF_wolb_rel2 << endl;cout <<" A_ovipos " << A_ovipos1 + A_ovipos_imm1 << " " << A_ovipos2 + A_ovipos_imm2<< endl;cout << " A_ovipos_wolb " <<A_ovipos_wolb1 + A_ovipos_wolb_imm1 << " " << A_ovipos_wolb2 + A_ovipos_wolb_imm2 << endl;cout <<" A_ovipos_wolb_rel " << A_ovipos_wolb_rel1 << " " << A_ovipos_wolb_rel2 << endl;
*/
	//shift A_ovipos and L
	for (int x=0; x<no_locns; x++){
	for (int i=t_steps-2; i>=0; i--) {
		A_ovipos[i+1][x] = A_ovipos[i][x];
		A_ovipos_wolb[i+1][x] = A_ovipos_wolb[i][x];
		A_ovipos_wolb_rel[i+1][x] = A_ovipos_wolb_rel[i][x];
		for (int c=0; c<no_classes; c++) {
			A_ovipos_imm[i+1+t_steps*x+t_steps*no_locns*c] = A_ovipos_imm[i+t_steps*x+t_steps*no_locns*c];
			A_ovipos_wolb_imm[i+1+t_steps*x+t_steps*no_locns*c] = A_ovipos_wolb_imm[i+t_steps*x+t_steps*no_locns*c];
			}
		L[i+1 +t_steps*x] = L[i+t_steps*x];
		L_wolb[i+1 +t_steps*x] = L_wolb[i+t_steps*x];
		}
//cout << "time1 " << time1 << " L[0+t_steps*locn] " << L[0+t_steps*x] << endl;
		L[0+t_steps*x]=0;
		L_wolb[0+t_steps*x]=0;
	}


}//end of time loop


freq_end = A_tot_wolb_in_rel/A_tot_in_rel;
cout << "freq_end " << freq_end << " A_tot_in_rel " << A_tot_in_rel << " A_tot_wolb_in_rel " << A_tot_wolb_in_rel;


return 0;			
	

}

//------------------------------"init_neighbours" function---------------------------------------
void init_neighbours(Matrix_Int& neighbrs, Matrix_Int& neighbrs_ne, vector<int>& no_neighbrs, vector<int>& no_neighbrs_ne){

for (int loc_no=0; loc_no<no_locns; loc_no++){
	for (int j=0; j<8; j++){
		neighbrs[loc_no][j]=0;
		neighbrs_ne[loc_no][j]=0;
	}
	no_neighbrs[loc_no]=0;
	no_neighbrs_ne[loc_no]=0;
}

}



//------------------------------"get_neighbours" function----------------------------------------
void get_neighbours(Matrix_Int& loc_pos, Matrix_Int& neighbrs, Matrix_Int& neighbrs_ne, vector<int>& no_neighbrs, vector<int>& no_neighbrs_ne){

int loc_no, count;

loc_no=0;

for (int y=0; y<no_locns_y; y++) {
	loc_pos[0][y] = loc_no;	
//cout << "x " << 0  << " y " << y << " loc_no " << loc_no << endl;	
	++loc_no;
}
for (int x=1; x<no_locns_x; x++){
	loc_pos[x][no_locns_y-1] =loc_no;
//cout << "x " << x  << " y " << no_locns_y-1 << " loc_no " << loc_no << endl;	
	++loc_no;
}
for (int y=no_locns_y-2; y>=0; y--){
	loc_pos[no_locns_x-1][y] = loc_no;
//cout << "x " << no_locns_x-1  << " y " << y << " loc_no " << loc_no << endl;	
	++loc_no;
}
for (int x=no_locns_x-2; x>0; x--){
	loc_pos[x][0] = loc_no;
//cout << "x " << x  << " y " << 0 << " loc_no " << loc_no << endl;	
	++loc_no;
}
for (int y=1; y<no_locns_y-1; y++){
	loc_pos[1][y] = loc_no;
//cout << "x " << 1  << " y " << y << " loc_no " << loc_no << endl;	
	++loc_no;
}
for (int x=2; x<no_locns_x-1; x++){
	loc_pos[x][no_locns_y-2] = loc_no;
//cout << "x " << x  << " y " << no_locns_y-2 << " loc_no " << loc_no << endl;	
	++loc_no;
}
for (int y=no_locns_y-3; y>0; y--){
	loc_pos[no_locns_x-2][y] = loc_no;
//cout << "x " << no_locns_x-2  << " y " << y << " loc_no " << loc_no << endl;	
	++loc_no;
}
for (int x=no_locns_x-3; x>1; x--){
	loc_pos[x][1] = loc_no;
//cout << "x " << x  << " y " << 1 << " loc_no " << loc_no << endl;	
	++loc_no;
}
//all others	
for (int x=2; x<no_locns_x-2; x++){
	for (int y=2; y<no_locns_y-2; y++){
		loc_pos[x][y] = loc_no;
//cout << "x " << x  << " y " << y << " loc_no " << loc_no << endl;	
		++loc_no;
	}
}

//bottom left corner 
loc_no=0;
no_neighbrs[loc_no]=3;
neighbrs[loc_no][0] = loc_pos[1][0]; neighbrs[loc_no][1] = loc_pos[0][1]; neighbrs[loc_no][2] = loc_pos[1][1];
++loc_no;

no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[0][0] ; neighbrs[loc_no][1] = loc_pos[1][0] ; neighbrs[loc_no][2] = loc_pos[1][1] ; neighbrs[loc_no][3] = loc_pos[0][2] ; neighbrs[loc_no][4] = loc_pos[1][2];
++loc_no;

for (int y=2; y<no_locns_y-2; y++) {
	no_neighbrs[loc_no]=5;
	neighbrs[loc_no][0] = loc_pos[0][y-1] ;neighbrs[loc_no][1] = loc_pos[1][y-1] ; neighbrs[loc_no][2] = loc_pos[1][y] ; neighbrs[loc_no][3] = loc_pos[0][y+1] ; neighbrs[loc_no][4] = loc_pos[1][y+1];
++loc_no;
}

//top left corner
no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[0][no_locns_y-3] ; neighbrs[loc_no][1] = loc_pos[1][no_locns_y-3] ; neighbrs[loc_no][2] = loc_pos[1][no_locns_y-2] ; neighbrs[loc_no][3] = loc_pos[0][no_locns_y-1] ; neighbrs[loc_no][4] = loc_pos[1][no_locns_y-1];
++loc_no;

no_neighbrs[loc_no]=3;
neighbrs[loc_no][0] = loc_pos[0][no_locns_y-2]; neighbrs[loc_no][1] = loc_pos[1][no_locns_y-2]; neighbrs[loc_no][2] = loc_pos[1][no_locns_y-1];
++loc_no;

no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[0][no_locns_y-1] ; neighbrs[loc_no][1] = loc_pos[0][no_locns_y-2] ; neighbrs[loc_no][2] = loc_pos[1][no_locns_y-2] ; neighbrs[loc_no][3] = loc_pos[2][no_locns_y-2] ; neighbrs[loc_no][4] = loc_pos[2][no_locns_y-1];
++loc_no;

for (int x=2; x<no_locns_x-2; x++){
no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[x-1][no_locns_y-1] ;neighbrs[loc_no][1] = loc_pos[x+1][no_locns_y-1] ; neighbrs[loc_no][2] = loc_pos[x-1][no_locns_y-2] ; neighbrs[loc_no][3] = loc_pos[x][no_locns_y-2] ; neighbrs[loc_no][4] = loc_pos[x+1][no_locns_y-2];
++loc_no;
}

//top right corner
no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[no_locns_x-3][no_locns_y-1] ; neighbrs[loc_no][1] = loc_pos[no_locns_x-1][no_locns_y-1] ; neighbrs[loc_no][2] = loc_pos[no_locns_x-3][no_locns_y-2] ; neighbrs[loc_no][3] = loc_pos[no_locns_x-2][no_locns_y-2] ; neighbrs[loc_no][4] = loc_pos[no_locns_x-1][no_locns_y-2];
++loc_no;

no_neighbrs[loc_no]=3;
neighbrs[loc_no][0] = loc_pos[no_locns_x-2][no_locns_y-1]; neighbrs[loc_no][1] = loc_pos[no_locns_x-2][no_locns_y-2]; neighbrs[loc_no][2] = loc_pos[no_locns_x-1][no_locns_y-2];
++loc_no;

no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[no_locns_x-1][no_locns_y-3] ; neighbrs[loc_no][1] = loc_pos[no_locns_x-2][no_locns_y-3] ; neighbrs[loc_no][2] = loc_pos[no_locns_x-2][no_locns_y-2] ; neighbrs[loc_no][3] = loc_pos[no_locns_x-1][no_locns_y-1] ; neighbrs[loc_no][4] = loc_pos[no_locns_x-2][no_locns_y-1];
++loc_no;

for (int y = no_locns_y-3; y>1; y--){
no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[no_locns_x-1][y-1] ;neighbrs[loc_no][1] = loc_pos[no_locns_x-2][y-1] ; neighbrs[loc_no][2] = loc_pos[no_locns_x-2][y] ; neighbrs[loc_no][3] = loc_pos[no_locns_x-2][y+1] ; neighbrs[loc_no][4] = loc_pos[no_locns_x-1][y+1];
++loc_no;
}

//bottom right corner
no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[no_locns_x-2][1] ; neighbrs[loc_no][1] = loc_pos[no_locns_x-1][0] ; neighbrs[loc_no][2] = loc_pos[no_locns_x-2][0] ; neighbrs[loc_no][3] = loc_pos[no_locns_x-2][2] ; neighbrs[loc_no][4] = loc_pos[no_locns_x-1][2];
++loc_no;

no_neighbrs[loc_no]=3;
neighbrs[loc_no][0] = loc_pos[no_locns_x-1][1]; neighbrs[loc_no][1] = loc_pos[no_locns_x-2][1]; neighbrs[loc_no][2] = loc_pos[no_locns_x-2][0];
++loc_no;

no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[no_locns_x-1][0] ; neighbrs[loc_no][1] = loc_pos[no_locns_x-3][0] ; neighbrs[loc_no][2] = loc_pos[no_locns_x-3][1] ; neighbrs[loc_no][3] = loc_pos[no_locns_x-2][1]; neighbrs[loc_no][4] = loc_pos[no_locns_x-1][1];
++loc_no;

for (int x=no_locns_x-3; x>1; x--){
no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[x-1][0] ;neighbrs[loc_no][1] = loc_pos[x+1][0] ; neighbrs[loc_no][2] = loc_pos[x-1][1] ; neighbrs[loc_no][3] = loc_pos[x][1] ; neighbrs[loc_no][4] = loc_pos[x+1][1];
++loc_no;
}

no_neighbrs[loc_no]=5;
neighbrs[loc_no][0] = loc_pos[0][0] ; neighbrs[loc_no][1] = loc_pos[0][1]; neighbrs[loc_no][2] = loc_pos[1][1] ; neighbrs[loc_no][3] = loc_pos[2][1]; neighbrs[loc_no][4] = loc_pos[2][0];
++loc_no;

//Inner boundary
//bottom left corner
no_neighbrs[loc_no]=8;
neighbrs[loc_no][0] = loc_pos[0][0] ; neighbrs[loc_no][1] = loc_pos[0][1]; neighbrs[loc_no][2] = loc_pos[0][2] ; neighbrs[loc_no][3] = loc_pos[1][2]; neighbrs[loc_no][4] = loc_pos[2][2];neighbrs[loc_no][5] = loc_pos[2][1]; neighbrs[loc_no][6] = loc_pos[2][0]; neighbrs[loc_no][7] = loc_pos[1][0];
++loc_no;

for (int y=2; y<no_locns_y-2; y++){
no_neighbrs[loc_no]=8;
neighbrs[loc_no][0] = loc_pos[1][y-1] ; neighbrs[loc_no][1] = loc_pos[0][y-1]; neighbrs[loc_no][2] = loc_pos[0][y] ; neighbrs[loc_no][3] = loc_pos[0][y+1]; neighbrs[loc_no][4] = loc_pos[1][y+1];neighbrs[loc_no][5] = loc_pos[2][y+1]; neighbrs[loc_no][6] = loc_pos[2][y]; neighbrs[loc_no][7] = loc_pos[2][y-1];
++loc_no;
}

//top left corner
no_neighbrs[loc_no]=8;
neighbrs[loc_no][0] = loc_pos[1][no_locns_y-3] ; neighbrs[loc_no][1] = loc_pos[0][no_locns_y-3]; neighbrs[loc_no][2] = loc_pos[0][no_locns_y-2] ; neighbrs[loc_no][3] = loc_pos[0][no_locns_y-1]; neighbrs[loc_no][4] = loc_pos[1][no_locns_y-1];neighbrs[loc_no][5] = loc_pos[2][no_locns_y-1]; neighbrs[loc_no][6] = loc_pos[2][no_locns_y-2]; neighbrs[loc_no][7] = loc_pos[2][no_locns_y-3];
++loc_no;

for (int x=2; x<no_locns_x-2; x++){
no_neighbrs[loc_no]=8;
neighbrs[loc_no][0] = loc_pos[x][no_locns_y-3] ; neighbrs[loc_no][1] = loc_pos[x-1][no_locns_y-3]; neighbrs[loc_no][2] = loc_pos[x-1][no_locns_y-2] ; neighbrs[loc_no][3] = loc_pos[x-1][no_locns_y-1]; neighbrs[loc_no][4] = loc_pos[x][no_locns_y-1];neighbrs[loc_no][5] = loc_pos[x+1][no_locns_y-1]; neighbrs[loc_no][6] = loc_pos[x+1][no_locns_y-2]; neighbrs[loc_no][7] = loc_pos[x+1][no_locns_y-3];
++loc_no;
}

//top right corner
no_neighbrs[loc_no]=8;
neighbrs[loc_no][0] = loc_pos[no_locns_x-2][no_locns_y-3] ; neighbrs[loc_no][1] = loc_pos[no_locns_x-3][no_locns_y-3]; neighbrs[loc_no][2] = loc_pos[no_locns_x-3][no_locns_y-2] ; neighbrs[loc_no][3] = loc_pos[no_locns_x-3][no_locns_y-1]; neighbrs[loc_no][4] = loc_pos[no_locns_x-2][no_locns_y-1];neighbrs[loc_no][5] = loc_pos[no_locns_x-1][no_locns_y-1]; neighbrs[loc_no][6] = loc_pos[no_locns_x-1][no_locns_y-2]; neighbrs[loc_no][7] = loc_pos[no_locns_x-1][no_locns_y-3];
++loc_no;

for (int y=no_locns_y-3; y>1; y--){
no_neighbrs[loc_no]=8;
neighbrs[loc_no][0] = loc_pos[no_locns_x-2][y-1] ; neighbrs[loc_no][1] = loc_pos[no_locns_x-3][y-1]; neighbrs[loc_no][2] = loc_pos[no_locns_x-3][y] ; neighbrs[loc_no][3] = loc_pos[no_locns_x-3][y+1]; neighbrs[loc_no][4] = loc_pos[no_locns_x-2][y+1];neighbrs[loc_no][5] = loc_pos[no_locns_x-1][y+1]; neighbrs[loc_no][6] = loc_pos[no_locns_x-1][y]; neighbrs[loc_no][7] = loc_pos[no_locns_x-1][y-1];
++loc_no;
}

//bottom right corner
no_neighbrs[loc_no]=8;
neighbrs[loc_no][0] = loc_pos[no_locns_x-2][0] ; neighbrs[loc_no][1] = loc_pos[no_locns_x-3][0]; neighbrs[loc_no][2] = loc_pos[no_locns_x-3][1]; neighbrs[loc_no][3] = loc_pos[no_locns_x-3][2]; neighbrs[loc_no][4] = loc_pos[no_locns_x-2][2];neighbrs[loc_no][5] = loc_pos[no_locns_x-1][2]; neighbrs[loc_no][6] = loc_pos[no_locns_x-1][1]; neighbrs[loc_no][7] = loc_pos[no_locns_x-1][0];
++loc_no;

for (int x=no_locns_x-3; x>1; x--){
no_neighbrs[loc_no]=8;
neighbrs[loc_no][0] = loc_pos[x][0] ; neighbrs[loc_no][1] = loc_pos[x-1][0]; neighbrs[loc_no][2] = loc_pos[x-1][1]; neighbrs[loc_no][3] = loc_pos[x-1][2]; neighbrs[loc_no][4] = loc_pos[x][2];neighbrs[loc_no][5] = loc_pos[x+1][2]; neighbrs[loc_no][6] = loc_pos[x+1][1]; neighbrs[loc_no][7] = loc_pos[x+1][0];
++loc_no;
}


//inside blocks
for (int x=2; x<no_locns_x-2; x++){
	for (int y=2; y<no_locns_y-2; y++){
	no_neighbrs[loc_no]=8;
	neighbrs[loc_no][0] = loc_pos[x-1][y-1] ; neighbrs[loc_no][1] = loc_pos[x][y-1]; neighbrs[loc_no][2] = loc_pos[x+1][y-1]; neighbrs[loc_no][3] = loc_pos[x-1][y]; neighbrs[loc_no][4] = loc_pos[x+1][y];neighbrs[loc_no][5] = loc_pos[x-1][y+1]; neighbrs[loc_no][6] = loc_pos[x][y+1]; neighbrs[loc_no][7] = loc_pos[x+1][y+1];
++loc_no;
	}
}


//make new neighbours arrays to exclude empty neighbours
for (int loc_no=0; loc_no<no_locns; loc_no++){
	count=0;
	for (int i=0; i<no_neighbrs[loc_no]; i++){
		if (loc_class[neighbrs[loc_no][i]]>0){
			neighbrs_ne[loc_no][count] = neighbrs[loc_no][i];
			++no_neighbrs_ne[loc_no];
			++count;
		}
	}
}

}

//---------------------------------"get_release_locns" function -----------------------------------
void get_release_locns(Matrix_Int& loc_ind, vector<int>& release_flag){ 
	for (int locn=0; locn<no_locns; locn++){
		
		//single central release
		if (loc_ind[locn][0]>0.5*(no_locns_x-1)-15 && loc_ind[locn][0]<0.5*(no_locns_x-1)+15 && loc_ind[locn][1]>0.5*(no_locns_y-1)-15 && loc_ind[locn][1]<0.5*(no_locns_y-1)+15) release_flag[locn]=1;	



		//if (loc_class[locn]==2) release_flag[locn]=1;
		
		/*if (loc_ind[locn][0]>32-3 && loc_ind[locn][0]<32+3){
			 if (loc_ind[locn][1]>32-3 && loc_ind[locn][1]<(32+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>57-3 && loc_ind[locn][1]<(57+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>82-3 && loc_ind[locn][1]<(82+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>107-3 && loc_ind[locn][1]<(107+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>132-3 && loc_ind[locn][1]<(132+3)) release_flag[locn]=1;
		}
		if (loc_ind[locn][0]>57-3 && loc_ind[locn][0]<57+3){
			 if (loc_ind[locn][1]>32-3 && loc_ind[locn][1]<(32+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>57-3 && loc_ind[locn][1]<(57+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>82-3 && loc_ind[locn][1]<(82+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>107-3 && loc_ind[locn][1]<(107+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>132-3 && loc_ind[locn][1]<(132+3)) release_flag[locn]=1;
		}
		if (loc_ind[locn][0]>82-3 && loc_ind[locn][0]<82+3){
			 if (loc_ind[locn][1]>32-3 && loc_ind[locn][1]<(32+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>57-3 && loc_ind[locn][1]<(57+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>82-3 && loc_ind[locn][1]<(82+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>107-3 && loc_ind[locn][1]<(107+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>132-3 && loc_ind[locn][1]<(132+3)) release_flag[locn]=1;
		}
		if (loc_ind[locn][0]>107-3 && loc_ind[locn][0]<107+3){
			 if (loc_ind[locn][1]>32-3 && loc_ind[locn][1]<(32+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>57-3 && loc_ind[locn][1]<(57+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>82-3 && loc_ind[locn][1]<(82+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>107-3 && loc_ind[locn][1]<(107+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>132-3 && loc_ind[locn][1]<(132+3)) release_flag[locn]=1;
		}
		if (loc_ind[locn][0]>132-3 && loc_ind[locn][0]<132+3){
			 if (loc_ind[locn][1]>32-3 && loc_ind[locn][1]<(32+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>57-3 && loc_ind[locn][1]<(57+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>82-3 && loc_ind[locn][1]<(82+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>107-3 && loc_ind[locn][1]<(107+3)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>132-3 && loc_ind[locn][1]<(132+3)) release_flag[locn]=1;
		}*/

		//larger blocks
		/*if (loc_ind[locn][0]>30-8 && loc_ind[locn][0]<30+8){
			 if (loc_ind[locn][1]>30-8 && loc_ind[locn][1]<(30+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>65-8 && loc_ind[locn][1]<(65+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>100-8 && loc_ind[locn][1]<(100+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>135-8 && loc_ind[locn][1]<(135+8)) release_flag[locn]=1;
		}
		if (loc_ind[locn][0]>30-8 && loc_ind[locn][0]<30+8){
			 if (loc_ind[locn][1]>30-8 && loc_ind[locn][1]<(30+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>65-8 && loc_ind[locn][1]<(65+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>100-8 && loc_ind[locn][1]<(100+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>135-8 && loc_ind[locn][1]<(135+8)) release_flag[locn]=1;
		}
		if (loc_ind[locn][0]>30-8 && loc_ind[locn][0]<30+8){
			 if (loc_ind[locn][1]>30-8 && loc_ind[locn][1]<(30+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>65-8 && loc_ind[locn][1]<(65+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>100-8 && loc_ind[locn][1]<(100+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>135-8 && loc_ind[locn][1]<(135+8)) release_flag[locn]=1;
		}
		if (loc_ind[locn][0]>30-8 && loc_ind[locn][0]<30+8){
			 if (loc_ind[locn][1]>30-8 && loc_ind[locn][1]<(30+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>65-8 && loc_ind[locn][1]<(65+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>100-8 && loc_ind[locn][1]<(100+8)) release_flag[locn]=1;
			 if (loc_ind[locn][1]>135-8 && loc_ind[locn][1]<(135+8)) release_flag[locn]=1;
		}
		*/


	}

}

//---------------------------------"dispersal_new" function---------------------------------------

void dispersal_new(Matrix_Int neighbrs, Matrix_Int neighbrs_ne, vector<int> no_neighbrs, vector<int> no_neighbrs_ne, Matrix_Double& A_vec, vector<double>& A_imm_vec, int age1, vector<int> lambda_class, int ta_steps){

vector<double> leave_loc(no_locns), leave_loc_emp(no_locns);
Matrix_Double leave_imm(no_locns, Row_Double(no_classes)), leave_empty(no_locns, Row_Double(no_classes));

for (int x=0; x<no_locns; x++){
	//if (loc_class[x]>0){
	leave_loc[x] = move_prob*A_vec[age1][x];
	A_vec[age1][x] -= leave_loc[x];
	//}
	//else {
	//leave_loc_emp[x] = move_prob_emp*A_vec[age1][x];
	//A_vec[age1][x] -= leave_loc_emp[x];
	//}
	for (int classes=0; classes<no_classes; classes++) {
//move the loop below up here for efficiency?
		//if (loc_class[x]>0) {
			leave_imm[x][classes] = move_prob*A_imm_vec[age1 + ta_steps*x + ta_steps*no_locns*classes];
			A_imm_vec[age1 + ta_steps*x + ta_steps*no_locns*classes]-=leave_imm[x][classes];
		//}
		//else {
		//	leave_empty[x][classes] = move_prob_emp*A_imm_vec[age1 + ta_steps*x + ta_steps*no_locns*classes];
		//	A_imm_vec[age1 + ta_steps*x + ta_steps*no_locns*classes] -= leave_empty[x][classes];
		//}
	}
}

for (int x=0; x<no_locns; x++){
	for (int classes=0; classes<no_classes; classes++){
		//if (loc_class[x]>0){
		for (int n=0; n<no_neighbrs[x]; n++){
		 	A_imm_vec[age1 + ta_steps*neighbrs[x][n] + ta_steps*no_locns*classes]+=leave_imm[x][classes]/no_neighbrs[x];
			if (classes == lambda_class[x]) A_imm_vec[age1 + ta_steps*neighbrs[x][n] + ta_steps*no_locns*classes] +=leave_loc[x]/no_neighbrs[x];
		}
		//}
		//if (loc_class[x]==0){
		//if (no_neighbrs_ne[x]>0){
		//for (int n=0; n<no_neighbrs_ne[x]; n++){
		// 	A_imm_vec[age1 + ta_steps*neighbrs_ne[x][n] + ta_steps*no_locns*classes]+=leave_empty[x][classes]/no_neighbrs_ne[x];
		//	if (classes == lambda_class[x]) A_imm_vec[age1 + ta_steps*neighbrs_ne[x][n] + ta_steps*no_locns*classes] +=leave_loc_emp[x]/no_neighbrs_ne[x];

		//}
		//}
		//else {
		//for (int n=0; n<no_neighbrs[x]; n++){
		// 	A_imm_vec[age1 + ta_steps*neighbrs[x][n] + ta_steps*no_locns*classes]+=leave_imm[x][classes]/no_neighbrs[x];
		//	if (classes == lambda_class[x]) A_imm_vec[age1 + ta_steps*neighbrs[x][n] + ta_steps*no_locns*classes] +=leave_loc[x]/no_neighbrs[x];
		//}
		//}
		//}
	}
}

}

//--------------------------"dispersal_new_simp" function-----------------------------------------
void dispersal_new_simp(Matrix_Int neighbrs, Matrix_Int neighbrs_ne, vector<int> no_neighbrs, vector<int> no_neighbrs_ne, Matrix_Double& A_vec, int age1){

vector<double> leave(no_locns), leave_empty(no_locns);

for (int x=0; x<no_locns; x++){
	//if (loc_class[x]>0) {
		leave[x] = move_prob*A_vec[age1][x];
		A_vec[age1][x]-=leave[x];
	//}
	//else {
	//	leave_empty[x] = move_prob_emp*A_vec[age1][x];
	//	A_vec[age1][x] -= leave_empty[x];
	//}
}

for (int x=0; x<no_locns; x++){
	//if (loc_class[x]>0){
	for (int n=0; n<no_neighbrs[x]; n++){
		 A_vec[age1][neighbrs[x][n]]+=leave[x]/no_neighbrs[x];
		}
	//}
	//if (loc_class[x]==0){
	//if (no_neighbrs_ne[x]>0){
	//for (int n=0; n<no_neighbrs_ne[x]; n++){
	//	 A_vec[age1][neighbrs[x][n]]+=leave_empty[x]/no_neighbrs_ne[x];
	//}
	//}
	//else {
	//for (int n=0; n<no_neighbrs[x]; n++){
	//	 A_vec[age1][neighbrs[x][n]]+=leave[x]/no_neighbrs[x];
	//	}

	//}
	//}

}

}

//--------------------------------"dispersal_new_simp2" function---------------------------------

void dispersal_new_simp2(Matrix_Int neighbrs, Matrix_Int neighbrs_ne, vector<int> no_neighbrs, vector<int> no_neighbrs_ne, vector<double>& A_vec){

vector<double> leave(no_locns), leave_empty(no_locns);

for (int x=0; x<no_locns; x++){
	//if (loc_class[x]>0) {
		leave[x] = move_prob*A_vec[x];
		A_vec[x]-=leave[x];
	//}
	//else {
	//	leave_empty[x] = move_prob_emp*A_vec[x];
	//	A_vec[x] -= leave_empty[x];
	//}
}

for (int x=0; x<no_locns; x++){
	//if (loc_class[x]>0){
	for (int n=0; n<no_neighbrs[x]; n++){
		 A_vec[neighbrs[x][n]]+=leave[x]/no_neighbrs[x];
	}
	//}
	//if (loc_class[x]==0){
	//if (no_neighbrs_ne[x]>0){
	//	for (int n=0; n<no_neighbrs_ne[x]; n++){
	//	 	A_vec[neighbrs[x][n]]+=leave_empty[x]/no_neighbrs_ne[x];
	//	}
	//}
	//else {
	//for (int n=0; n<no_neighbrs[x]; n++){
	//	 A_vec[neighbrs[x][n]]+=leave[x]/no_neighbrs[x];
	//}

	//}
	//}
}

}

//-------------------------------------"write_loc_indices" function-------------------------------
void write_loc_indices(ofstream& loc_ind_out, Matrix_Int& loc_ind){
int loc_no;

loc_no=0;
for (int y=0; y<no_locns_y; y++) {
loc_ind_out <<  0  <<" "<<  y <<" "<<  loc_no << endl;
loc_ind[loc_no][0]=0; loc_ind[loc_no][1]=y;	
	++loc_no;
}
for (int x=1; x<no_locns_x; x++){
loc_ind_out <<  x  <<" "<<   no_locns_y-1 <<" "<<   loc_no << endl;
loc_ind[loc_no][0]=x; loc_ind[loc_no][1]=no_locns_y-1;		
	++loc_no;
}
for (int y=no_locns_y-2; y>=0; y--){
loc_ind_out <<  no_locns_x-1  <<" "<<   y <<" "<<   loc_no << endl;
loc_ind[loc_no][0]=no_locns_x-1; loc_ind[loc_no][1]=y;		
	++loc_no;
}
for (int x=no_locns_x-2; x>0; x--){
loc_ind_out <<  x  <<" "<<   0 <<" "<<   loc_no << endl;
loc_ind[loc_no][0]=x; loc_ind[loc_no][1]=0;		
	++loc_no;
}
for (int y=1; y<no_locns_y-1; y++){
loc_ind_out <<  1  <<" "<<   y <<" "<<   loc_no << endl;
loc_ind[loc_no][0]=1; loc_ind[loc_no][1]=y;		
	++loc_no;
}
for (int x=2; x<no_locns_x-1; x++){
loc_ind_out <<  x  <<" "<<   no_locns_y-2 <<" "<<   loc_no << endl;
loc_ind[loc_no][0]=x; loc_ind[loc_no][1]=no_locns_y-2;		
	++loc_no;
}
for (int y=no_locns_y-3; y>0; y--){
loc_ind_out <<  no_locns_x-2  <<" "<<   y <<" "<<   loc_no << endl;
loc_ind[loc_no][0]=no_locns_x-2; loc_ind[loc_no][1]=y;		
	++loc_no;
}
for (int x=no_locns_x-3; x>1; x--){
loc_ind_out <<  x  <<" "<<   1 << " "<<  loc_no << endl;
loc_ind[loc_no][0]=x; loc_ind[loc_no][1]=1;		
	++loc_no;
}
//all others	
for (int x=2; x<no_locns_x-2; x++){
	for (int y=2; y<no_locns_y-2; y++){
loc_ind_out <<  x  <<" "<<   y <<" "<<   loc_no << endl;
loc_ind[loc_no][0]=x; loc_ind[loc_no][1]=y;		
		++loc_no;
	}
}

}


//----------------------------------function "shift_vec3D"------------------------------------------
void shift_vec3D(vector_3D the_vec, int the_size1, int the_size2, int the_size3){

for (int i=0; i<the_size1-1; i++){
	for (int j=0; j<the_size2; j++){
		for (int k=0; k<the_size3; k++){
			//cout << "i " << i << " j " << j << " k " << k << endl;
			the_vec[i+1][j][k] = the_vec[i][j][k];
		}
	}
}

the_vec[0][the_size2-1][the_size3-1] = 0;

}



