// Copyright © 2020 Christina Daniel. All rights reserved.
// First, set paths by typing in terminal:
// MKLPATH=/opt/intel/oneapi/mkl/latest/lib
// MKLINCLUDE=/opt/intel/oneapi/mkl/latest/include
// MKLROOT=/opt/intel/oneapi/mkl/latest
// icpc -std=c++11 -L${MKLROOT}/lib -I${MKLROOT}/include $MKLPATH/libmkl_intel_ilp64.a $MKLPATH/libmkl_sequential.a $MKLPATH/libmkl_core.a $MKLPATH/libmkl_intel_ilp64.a $MKLPATH/libmkl_sequential.a $MKLPATH/libmkl_core.a -DMKL_ILP64 -lpthread -o a.out main.cpp
// To run: ./a.out


/* define version string */
static char _V_[] = "@(#)newton Real.cpp 01.01 -- Copyright (C) Future Team Aps";
#include <time.h>
#include <float.h>
#define MAXITER 50

#include "/Users/christinadaniel/Desktop/Christina_Desktop/project/project/include/briefsymbolicc++.h"

using namespace std;
#include <mkl.h>
#include "mkl_lapacke.h"
#include <iomanip>
#include <vector>
#include <complex>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <limits>
#include <math.h>
#include <cmath>
#include <unordered_map>
#include <deque>
#include <algorithm>
typedef std::numeric_limits< double > dbl;

int up = 1;
int down = -1;
const double pi = 3.1415926535897932384626433832795028841971693993751;
complex<double> minus_one(-1.0,0.0);
complex<double> my_zero(0.0,0.0);
complex<double> my_one(1.0,0.0);
complex<double> minus_i(0.0,-1.0);
complex<double> plus_i(0.0,1.0);
complex<double> negative_pi_times_i(0.0,-pi);
complex<double> negative_two_pi_times_i(0.0,-2*pi);
complex<double> two_pi_times_i(0.0,2*pi);
vector<vector<int> > combinations_up;
vector<vector<int> > combinations_down;
unordered_map< string, vector<int> > map_for_up_spin_sets;
unordered_map< string, complex<double> > map_for_up_spin_signs;
unordered_map< string, vector<int> > map_for_down_spin_sets;
unordered_map< string, complex<double> > map_for_down_spin_signs;
unordered_map< string, complex<double> > map_for_KE_coefficient;
unordered_map< string, complex<double> > map_for_PE_coefficient;
unordered_map< string, vector<int> > map_for_PE_up_sets;
unordered_map< string, int > map_for_KE_up_ket_to_index_of_up_ket;
unordered_map< string, int > map_for_KE_down_ket_to_index_of_down_ket;
unordered_map< string, int > map_for_PE_up_ket_to_index_of_up_ket;
unordered_map< string, int > map_for_PE_down_ket_to_index_of_down_ket;

double V_eff = 2.0;
complex<double> t_0(0.0, 0.0);
complex<double> t_1(1.0, 0.0);
complex<double> t_2(0.0, 0.0);
bool allow_second_neighbor_hopping = true;
bool print = false;
int V = 4; // number of atoms

const int number_down = 6; // Example: 4 lattice sites, 2 up electrons, 2 down electrons --> 4 choose 2 = 6
const int num_up_for_N = 6; // Example: 4 lattice sites, 2 up electrons, 2 down electrons --> 4 choose 2 = 6
const int num_up_for_N_minus_1 = 4; // Example: 4 lattice sites, 1 up electron, 2 down electrons --> 4 choose 1 = 4
const int num_up_for_N_plus_1 = 4; // Example: 4 lattice sites, 3 up electrons, 2 down electrons --> 4 choose 3 = 4

const int D_N = num_up_for_N*number_down;
const int D_N_MINUS_ONE = num_up_for_N_minus_1*number_down;
const int D_N_PLUS_ONE = num_up_for_N_plus_1*number_down;
const unsigned long long int array_size_for_N = D_N*D_N;
const unsigned long long int array_size_for_N_minus_1 = D_N_MINUS_ONE*D_N_MINUS_ONE;
const unsigned long long int array_size_for_N_plus_1 = D_N_PLUS_ONE*D_N_PLUS_ONE;
deque<complex<double> > kinetic;
deque<complex<double> > potential;
double* hamiltonian_N;
double* hamiltonian_N_minus_one;
double* hamiltonian_N_plus_one;
double* z_N; // eigenvectors for N electrons
double* z_N_minus_one; // eigenvectors for N-1 electrons
double* z_N_plus_one; // eigenvectors for N+1 electrons
double exact_ground_state[D_N];
double* ground_state_N_minus_one;
double* ground_state_N_plus_one;
double w_N[D_N]; // eigenvalues for N electrons
double w_N_minus_one[D_N_MINUS_ONE]; // eigenvalues for N-1 electrons
double w_N_plus_one[D_N_PLUS_ONE]; // eigenvalues for N+1 electrons
long long int ifail_N[D_N];
long long int ifail_N_minus_one[D_N_MINUS_ONE];
long long int ifail_N_plus_one[D_N_PLUS_ONE];
long long int GROUND_STATE_INDEX;
// Green's function parameters
int sigma = up;

vector<double> times;
const int number_of_time_points = 4096;
double time_step = 0.02442002442002442;
double start_time = 0;

vector<double> omegas_greater;
vector<double> omegas_lesser;

vector<long double> omegas_n_plus_one;
vector<long double> omegas_n_minus_one;

double selected_omegas_part1[D_N_PLUS_ONE];
double selected_omegas_part2[D_N_MINUS_ONE];
vector<double> temperature_array;

complex<double> g1[number_of_time_points];
double g1_step_1_vector[D_N_PLUS_ONE];
double g1_step_2_vector[D_N_PLUS_ONE];
complex<double> g1_step_3_vector[D_N_PLUS_ONE];
complex<double> g1_step_4_vector[D_N_PLUS_ONE];
complex<double> g1_step_5_vector[D_N];
complex<double> g1_step_6_vector[D_N];
complex<double> g1_step_7_vector[D_N];
complex<double> g1_step_8_vector[D_N];
complex<double> g2[number_of_time_points];
double step_1_vector[D_N];
complex<double> step_2_vector[D_N];
complex<double> step_3_vector[D_N];
complex<double> step_4_vector[D_N_MINUS_ONE];
complex<double> step_5_vector[D_N_MINUS_ONE];
complex<double> step_6_vector[D_N_MINUS_ONE];
complex<double> step_7_vector[D_N_MINUS_ONE];
complex<double> step_8_vector[D_N];
// temperature-dependent green's function
// double state[D_N];
// *** retarded green's function ***
complex<double> g_R_part_1[number_of_time_points];
complex<double> g_R_part_2[number_of_time_points];
// part 1
double* matrix_n_c_l; // <n|c|l>
double* matrix_l_c_dagger_n; // <l|c-dagger|n>
unordered_map< int, complex<double> > map_for_c_signs_PART_1;
unordered_map< int, int > map_for_c_on_ket_to_index_PART_1;
unordered_map< int, complex<double> > map_for_c_dagger_signs_PART_1;
unordered_map< int, int > map_for_c_dagger_on_ket_to_index_PART_1;
// part 2
double* matrix_m_c_n; // <m|c|n>
double* matrix_n_c_dagger_m; // <n|c-dagger|m>
unordered_map< int, complex<double> > map_for_c_signs;
unordered_map< int, int > map_for_c_on_ket_to_index;
unordered_map< int, complex<double> > map_for_c_dagger_signs;
unordered_map< int, int > map_for_c_dagger_on_ket_to_index;
// Frequency Domain, Lehmann, Ground State Only
vector<complex<double>> g_omega_exact_greater;
vector<complex<double>> g_omega_exact_lesser;
vector<long double> matrix_elements_n_plus_one;
vector<long double> matrix_elements_n_minus_one;
// PARTIAL SET FUNCTIONS
void combinations_partial(int v[], int start, int n, int k, int maxk, bool spin_up) {
    int i;
    vector<int> set;
    
    if (k > maxk) {
        set.clear();
        for (i=1; i<=maxk; i++) {
            set.push_back(v[i]-1);
        }
        if (spin_up == true) {
            combinations_up.push_back(set);
        }
        else {
            combinations_down.push_back(set);
        }
        return;
    }
    
    for (i=start; i<=n; i++) {
        
        v[k] = i;
        combinations_partial(v, i+1, n, k+1, maxk, spin_up);
    }
}
vector<int> partial_set_in_bit_form(const vector<int>& set) {
    vector<int> zeros;
    for(int i = 0; i < V; i++) {
        zeros.push_back(0);
    }
    for(int i = 0; i < set.size(); i++) {
        zeros[set[i]] = 1;
    }
    return zeros;
}
vector<vector<int> > get_up_spin_sets_in_bit_form() {
    vector<vector<int> > up_spin_sets;
    long num_up = combinations_up.size();
    for (int i = 0; i < num_up; i++) {
        up_spin_sets.push_back(partial_set_in_bit_form(combinations_up[i]));
    }
    return up_spin_sets;
}
vector<vector<int> > get_down_spin_sets_in_bit_form() {
    vector<vector<int> > down_spin_sets;
    long num_down = combinations_down.size();
    for (int i = 0; i < num_down; i++) {
        down_spin_sets.push_back(partial_set_in_bit_form(combinations_down[i]));
    }
    return down_spin_sets;
}
vector<int> apply_c_dagger_special(int k_index, const vector<int>& input, complex<double>& coefficient, bool is_up_section, int N_up_to_hop_over) {
    for (int i = 0; i < V; i++) {
        if ((k_index==i) && (input[i]==1)) {
            coefficient = my_zero;
            return input;
        }
    }
    int jumps = 0;
    for(int j = 0; j < k_index; j++) {
        if (input[j] == 1) {
            jumps++;
        }
    }
    if (is_up_section == false) {
        jumps = jumps + N_up_to_hop_over;
    }
    if ( jumps%2 != 0 ) {
        coefficient = coefficient * minus_one;
    }
    
    vector<int> output = input;
    output[k_index] = 1;
    return output;
}
vector<int> apply_c_special(int k_index, const vector<int>& input, complex<double>& coefficient, bool is_up_section, int N_up_to_hop_over) {
    for (int i = 0; i < V; i++) {
        if ((k_index==i) && (input[i]==1)) {
            int jumps = 0;
            for (int j = 0; j < k_index; j++) {
                if (input[j] == 1) {
                    jumps++;
                }
            }
            if (is_up_section == false) {
                jumps = jumps + N_up_to_hop_over;
            }
            if ( jumps%2 != 0 ) {
                coefficient = coefficient * minus_one;
            }
            vector<int> output = input;
            output[k_index] = 0;
            return output;
        }
    }
    coefficient = my_zero;
    return input;
}
// DIAGONALIZATION WITH PARTIAL SETS
void fill_maps(int N_up, int N_down) {
    
    cout << "allocating hopping matrix..." << endl;
    //coefficients for KE
    complex<double> T_matrix[V][V];
    if (allow_second_neighbor_hopping == true) {
       for (int row = 0; row < V; row++) {
            for (int col = 0; col < V; col++) {
                 if ( abs(row-col) == 0 ) { // 4-Site Hydrogen Ring
                    T_matrix[row][col] = t_0;
                 }
                 else if ( abs(row-col) == 1 ) {
                    T_matrix[row][col] = t_1;
                 }
                 else if ( abs(row-col) == (V-1) ) {
                    T_matrix[row][col] = t_1;
                 }
                 else if ( abs(row-col) == 2 ) {
                      T_matrix[row][col] = t_2;
                 }
                 else {
                      T_matrix[row][col] = my_zero;
                 }
            }
       }
    }
    else {
        for (int row = 0; row < V; row++) {
            for (int col = 0; col < V; col++) {
                if ( abs(row-col) == 1) { // nearest-neighbor hopping
                   T_matrix[row][col] = t_1;
                }
                else if ( abs(row-col) == V-1) { // periodic boundary condition
                   T_matrix[row][col] = t_1;
                }
                else {
                   T_matrix[row][col] = my_zero;
                }
            }
        }
    }
    cout << "allocating KE coefficients..." << endl;
    double scale = 2 * pi/V;
    string coeff_key;
    for (int i = 0; i < V; i++) {
        for(int j = 0; j < V; j++) {
            complex<double> a_factor(real(T_matrix[i][j]), 0.0);
            coeff_key = "_" + to_string(i) + "_" + to_string(j) + "_";
            map_for_KE_coefficient.insert(make_pair(coeff_key,a_factor));
        }
    }
    
    cout << "allocating up data..." << endl;
    // sets for up-spins
    string up_key;
    vector<int> temp_ket_up;
    vector<int> final_ket_up;
    complex<double> coefficient_up;
    vector<vector<int> > up_spin_sets_in_bit_form = get_up_spin_sets_in_bit_form();
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            for (int set_index = 0; set_index < up_spin_sets_in_bit_form.size(); set_index++) {
                coefficient_up = my_one;
                temp_ket_up = apply_c_special(j, up_spin_sets_in_bit_form[set_index], coefficient_up, true, 0);
                final_ket_up = apply_c_dagger_special(i, temp_ket_up, coefficient_up, true, 0);
                up_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(set_index) +"_";
                map_for_up_spin_sets.insert(make_pair(up_key,final_ket_up));
                map_for_up_spin_signs.insert(make_pair(up_key,coefficient_up));
            }
        }
    }
    cout << "allocating down data..." << endl;
    // sets for down-spins
    string down_key;
    vector<int> temp_ket_down;
    vector<int> final_ket_down;
    complex<double> coefficient_down;
    vector<vector<int> > down_spin_sets_in_bit_form = get_down_spin_sets_in_bit_form();
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            for (int set_index = 0; set_index < down_spin_sets_in_bit_form.size(); set_index++) {
                coefficient_down = my_one;
                temp_ket_down = apply_c_special(j, down_spin_sets_in_bit_form[set_index], coefficient_down, false, N_up);
                final_ket_down = apply_c_dagger_special(i, temp_ket_down, coefficient_down, false, N_up);
                down_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(set_index)+"_";
                map_for_down_spin_sets.insert(make_pair(down_key,final_ket_down));
                map_for_down_spin_signs.insert(make_pair(down_key,coefficient_down));
            }
        }
    }
    cout << "allocating PE data..." << endl;
    // PE
    complex<double> coefficient;
    vector<int> third_temp_ket_PE;
    vector<int> final_ket_up_PE;
    complex<double> scale_factor(V_eff,0.0);
    for (int up_index = 0; up_index < up_spin_sets_in_bit_form.size(); up_index++) {
        for(int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                coefficient = scale_factor;
                third_temp_ket_PE = apply_c_special(i, up_spin_sets_in_bit_form[up_index], coefficient, true, 0);
                final_ket_up_PE = apply_c_dagger_special(j, third_temp_ket_PE, coefficient, true, 0);
                map_for_PE_coefficient.insert(make_pair("_"+ to_string(i) +"_"+ to_string(j) +"_"+ to_string(up_index)+"_",coefficient));
                map_for_PE_up_sets.insert(make_pair("_"+ to_string(i) +"_"+ to_string(j) +"_"+ to_string(up_index)+"_",final_ket_up_PE));
            }
        }
    }
}
void convert_from_ket_to_index_of_part(int N_up, int N_down) {
    vector<vector<int> > up_spin_sets_in_bit_form = get_up_spin_sets_in_bit_form();
    vector<vector<int> > down_spin_sets_in_bit_form = get_down_spin_sets_in_bit_form();
    long num_up = combinations_up.size();
    long num_down = combinations_down.size();
    cout << "converting up..." << endl;
    // sets for up-spins
    string up_key;
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            for (int set_index = 0; set_index < num_up; set_index++) {
                up_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(set_index)+"_";
                for (int integer = 0; integer < num_up; integer++) {
                    if (map_for_up_spin_sets.find(up_key)->second == up_spin_sets_in_bit_form[integer]) {
                        map_for_KE_up_ket_to_index_of_up_ket.insert(make_pair(up_key,integer));
                        break;
                    }
                }
            }
        }
    }
    
    
    cout << "converting down..." << endl;
    // sets for down-spins
    string down_key;
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            for (int set_index = 0; set_index < num_down; set_index++) {
                down_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(set_index)+"_";
                for (int integer = 0; integer < num_down; integer++) {
                    if (map_for_down_spin_sets.find(down_key)->second == down_spin_sets_in_bit_form[integer] ) {
                        map_for_KE_down_ket_to_index_of_down_ket.insert(make_pair(down_key,integer));
                        map_for_PE_down_ket_to_index_of_down_ket.insert(make_pair(down_key,integer));
                        break;
                    }
                }
            }
        }
    }
    
    
    // PE
    cout << "converting PE data..." << endl;
    string PE_key;
    for (int up_index = 0; up_index < num_up; up_index++) {
        for(int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                PE_key = "_"+ to_string(i) +"_"+ to_string(j) +"_"+ to_string(up_index)+"_";
                for (int integer = 0; integer < num_up; integer++) {
                    if (map_for_PE_up_sets.find(PE_key)->second == up_spin_sets_in_bit_form[integer]) {
                        map_for_PE_up_ket_to_index_of_up_ket.insert(make_pair(PE_key,integer));
                        break;
                    }
                }
            }
        }
    }
    cout << "maps are set." << endl;
}
void matrix_for_T(int N_up, int N_down) {
    vector<vector<int> > up_spin_sets_in_bit_form = get_up_spin_sets_in_bit_form();
    vector<vector<int> > down_spin_sets_in_bit_form = get_down_spin_sets_in_bit_form();
    long num_up = combinations_up.size();
    long num_down = combinations_down.size();
    long D = num_up*num_down;
    complex<double> coefficient;
    complex<double> past_value;
    string coeff_key;
    string up_key;
    string down_key;
    int y;
    int z;
    int w;
    int x;
    long skip;
    
    if (num_up >= num_down) {
        skip = num_up;
        for(int i = 0; i < V; i++) {
            for(int j = 0; j < V; j++) {
                coeff_key = "_"+to_string(i) +"_"+ to_string(j) +"_";
                coefficient = map_for_KE_coefficient.find(coeff_key)->second;
                if (coefficient != my_zero) {
                    for (w = 0; w < num_up; w++) { // defines col or ket
                        for (x = 0; x < num_down; x++) { // defines col or ket
                            /* --- spin-up operator part --- */
                            coefficient = map_for_KE_coefficient.find(coeff_key)->second;
                            up_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(w)+"_";
                            coefficient = coefficient * map_for_up_spin_signs.find(up_key)->second;
                            if (coefficient != my_zero) {
                                y = map_for_KE_up_ket_to_index_of_up_ket.find(up_key)->second; // spin-up part of output changes
                                z = x; // spin-down part of output ket remains the same as that in input ket
                                kinetic[(x*skip+w)*D + z*skip+y] = kinetic[(x*skip+w)*D + z*skip+y] + coefficient;
                            }
                            /* --- spin-down operator part --- */
                            coefficient = map_for_KE_coefficient.find(coeff_key)->second;
                            down_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(x)+"_";
                            coefficient = coefficient * map_for_down_spin_signs.find(down_key)->second;
                            if (coefficient != my_zero) {
                                y = w; // spin-up part of output ket remains the same as that in input ket
                                z = map_for_KE_down_ket_to_index_of_down_ket.find(down_key)->second; // spin-down part of output changes
                                kinetic[(x*skip+w)*D + z*skip+y] = kinetic[(x*skip+w)*D + z*skip+y] + coefficient;
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        skip = num_down;
        for(int i = 0; i < V; i++) {
            for(int j = 0; j < V; j++) {
                coeff_key = "_"+to_string(i) +"_"+ to_string(j) +"_";
                coefficient = map_for_KE_coefficient.find(coeff_key)->second;
                if (coefficient != my_zero) {
                    for (w = 0; w < num_up; w++) { // defines col or ket
                        for (x = 0; x < num_down; x++) { // defines col or ket
                            /* --- spin-up operator part --- */
                            coefficient = map_for_KE_coefficient.find(coeff_key)->second;
                            up_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(w)+"_";
                            coefficient = coefficient * map_for_up_spin_signs.find(up_key)->second;
                            if (coefficient != my_zero) {
                                y = map_for_KE_up_ket_to_index_of_up_ket.find(up_key)->second; // spin-up part of output changes
                                z = x; // spin-down part of output ket remains the same as that in input ket
                                kinetic[(w*skip+x)*D + y*skip+z] = kinetic[(w*skip+x)*D + y*skip+z] + coefficient;
                            }
                            /* --- spin-down operator part --- */
                            coefficient = map_for_KE_coefficient.find(coeff_key)->second;
                            down_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(x)+"_";
                            coefficient = coefficient * map_for_down_spin_signs.find(down_key)->second;
                            if (coefficient != my_zero) {
                                y = w; // spin-up part of output ket remains the same as that in input ket
                                z = map_for_KE_down_ket_to_index_of_down_ket.find(down_key)->second; // spin-down part of output changes
                                kinetic[(w*skip+x)*D + y*skip+z] = kinetic[(w*skip+x)*D + y*skip+z] + coefficient;
                            }
                        }
                    }
                }
            }
        }
    }
}
void matrix_for_U(int N_up, int N_down) {
    long num_up = combinations_up.size();
    long num_down = combinations_down.size();
    long D = num_up * num_down;
    vector<vector<int> > up_spin_sets_in_bit_form = get_up_spin_sets_in_bit_form();
    vector<vector<int> > down_spin_sets_in_bit_form = get_down_spin_sets_in_bit_form();
    complex<double> coefficient;
    complex<double> past_value;
    string down_key;
    string PE_key;
    int y;
    int z;
    int w;
    int x;
    long skip;
    if (num_up >= num_down) {
        skip = num_up;
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (i == j) {
                    for (x = 0; x < num_down; x++) { // defines col or ket
                        down_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(x)+"_";
                        if (map_for_down_spin_signs.find(down_key)->second != my_zero) {
                            for (w = 0; w < num_up; w++) { // defines col or ket
                                PE_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(w)+"_";
                                coefficient = map_for_down_spin_signs.find(down_key)->second * map_for_PE_coefficient.find(PE_key)->second;
                                if (coefficient != my_zero) {
                                    y = map_for_PE_up_ket_to_index_of_up_ket.find(PE_key)->second;
                                    z = map_for_PE_down_ket_to_index_of_down_ket.find(down_key)->second;
                                    potential[(x*skip+w)*D + z*skip+y] = potential[(x*skip+w)*D + z*skip+y] + coefficient;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        skip = num_down;
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (i == j) {
                    for (x = 0; x < num_down; x++) { // defines col or ket
                        down_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(x)+"_";
                        if (map_for_down_spin_signs.find(down_key)->second != my_zero) {
                            for (w = 0; w < num_up; w++) { // defines col or ket
                                PE_key = "_"+to_string(i) +"_"+ to_string(j) +"_"+ to_string(w)+"_";
                                coefficient = map_for_down_spin_signs.find(down_key)->second * map_for_PE_coefficient.find(PE_key)->second;
                                if (coefficient != my_zero) {
                                    y = map_for_PE_up_ket_to_index_of_up_ket.find(PE_key)->second;
                                    z = map_for_PE_down_ket_to_index_of_down_ket.find(down_key)->second;
                                    potential[(w*skip+x)*D + y*skip+z] = potential[(w*skip+x)*D + y*skip+z] + coefficient;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
double min_value_double(vector<double> a) {
    
    double min = a[0];
    for (int i = 0; i < a.size(); i++) {
        if (a[i] <= min) {
            min = a[i];
        }
    }
    return min;
    
}
int min_value_int(vector<int> a) {
    
    int min = a[0];
    for (int i = 0; i < a.size(); i++) {
        if (a[i] <= min) {
            min = a[i];
        }
    }
    return min;
}
void add_matrices(int D) {
    unsigned long long int array_size;
    if (D == D_N) {
        array_size = array_size_for_N;
    }
    if (D == D_N_MINUS_ONE) {
        array_size = array_size_for_N_minus_1;
    }
    if (D == D_N_PLUS_ONE) {
        array_size = array_size_for_N_plus_1;
    }
    cout << "array_size in add function is: " << array_size << endl;
    int row_counter = 0;
    for (long i = 0; i < array_size; i++) {
        if (i%D <= row_counter) {
            kinetic[i] = kinetic[i]  + potential[i];
        }
        else {
            kinetic[i] = my_zero;
        }
        if ( (i+1)%D == 0) {
            row_counter++;
        }
    }
}
void convert(int D,double* hamiltonian) {
    unsigned long long int array_size;
    if (D == D_N) {
        array_size = array_size_for_N;
    }
    if (D == D_N_MINUS_ONE) {
        array_size = array_size_for_N_minus_1;
    }
    if (D == D_N_PLUS_ONE) {
        array_size = array_size_for_N_plus_1;
    }
    cout << "array_size in convert function is: " << array_size << endl;
    for (int i = 0; i < array_size; i++) {
        hamiltonian[i] = real(kinetic[i]);
    }
}
void diagonalize(int D, double* ground_state, double* hamiltonian, double* z, double* w, long long int* ifail) {
    
    int matrix_layout = LAPACK_ROW_MAJOR;
    int dim = D;
    int lda = D;
    int ldz = D;
    int il;
    int iu;
    int info;
    long long int m;
    double abstol = -1;
    double vl = 0.0;
    double vu = 100.0;
    cout << "calling LAPACK..." << endl;
    info = LAPACKE_dsyevr(matrix_layout,'V','A','L', dim, hamiltonian, lda, vl, vu, il, iu, abstol, &m, w, z, ldz,ifail);
    if (info != 0) {
        std::cout << "diagonalization failed" << std::endl;
        exit(1);
    }
    cout << "diagonalization succeeded" << endl;
    
    if (D == D_N) {
        double ground_state_energy = *min_element(w,w+D);
        cout << "The ground state energy for N electrons is " << ground_state_energy << endl;
        for (int j = 0; j < D; j++) {
            if (w[j] == ground_state_energy) {
                GROUND_STATE_INDEX = j;
                for (int i = 0; i < D; i++) {
                    ground_state[i] = z[i*D + j]; // ith element of jth eigenvector
                }
            }
        }
    }
}
void clear_all_maps() {
    map_for_up_spin_sets.clear();
    map_for_up_spin_signs.clear();
    map_for_down_spin_sets.clear();
    map_for_down_spin_signs.clear();
    map_for_KE_coefficient.clear();
    map_for_PE_coefficient.clear();
    map_for_PE_up_sets.clear();
    map_for_KE_up_ket_to_index_of_up_ket.clear();
    map_for_KE_down_ket_to_index_of_down_ket.clear();
    map_for_PE_up_ket_to_index_of_up_ket.clear();
    map_for_PE_down_ket_to_index_of_down_ket.clear();
}
// GREEN CALCULATION



// zero temperature (i.e. ground state only)
void green_part_1(int k_index_unprimed, int k_prime_index, double* ground_state, int N_up_original, int N_down_original) {
    
    cout << "obtaining up-spin sets..." << endl;
    combinations_up.clear();
    int array_one[100];
    int n = V;
    int k = N_up_original;
    combinations_partial(array_one, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_4 = get_up_spin_sets_in_bit_form();
    combinations_up.clear();
    int array_three[100];
    n = V;
    k = N_up_original + 1;
    combinations_partial(array_three, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_5 = get_up_spin_sets_in_bit_form();
    long num_down = combinations_down.size();

    for (int i = 0; i < times.size(); i++) {
        g1[i] = my_zero;
    }

    for (int i = 0; i < D_N_PLUS_ONE; i++) {
        g1_step_1_vector[i] = 0.0;
        g1_step_2_vector[i] = 0.0;
    }
    
    // STEP 1
    cout << "starting step 1..." << endl;
    int skip_l;
    int skip_n;
    if (up_spin_sets_in_bit_form_for_N_equals_5.size() >= num_down) {
       skip_l = up_spin_sets_in_bit_form_for_N_equals_5.size();
    }
    else {
         skip_l = num_down;
    }
    if (up_spin_sets_in_bit_form_for_N_equals_4.size() >= num_down) {
       skip_n = up_spin_sets_in_bit_form_for_N_equals_4.size();
    }
    else {
         skip_n = num_down;
    }
    for (int w = 0; w < num_up_for_N; w++) {
        for(int x = 0; x < num_down; x++) {
            complex<double> coefficient(ground_state[x*skip_n+w], 0.0);
            vector<int> output = apply_c_dagger_special(k_prime_index, up_spin_sets_in_bit_form_for_N_equals_4[w], coefficient, true, 0);
            for (int y = 0; y < num_up_for_N_plus_1; y++) { // access |N+1>-basis
                if (output == up_spin_sets_in_bit_form_for_N_equals_5[y]) {
                    int z = x; // down-set does not change
                    g1_step_1_vector[y*skip_l+z] = g1_step_1_vector[y*skip_l+z] + real(coefficient); // case 2 for indexing
                }
            }
        }
    }
    // STEP 2
    cout << "starting step 2..." << endl;
    for (int eigenvector_index = 0; eigenvector_index < D_N_PLUS_ONE; eigenvector_index++) {
        for (int element = 0; element < D_N_PLUS_ONE; element++) {
            g1_step_2_vector[eigenvector_index] = g1_step_2_vector[eigenvector_index] + z_N_plus_one[element*D_N_PLUS_ONE+eigenvector_index]*g1_step_1_vector[element];
        }
    }
    for (int t_index = 0; t_index < times.size(); t_index++) {
        for (int i = 0; i < D_N_PLUS_ONE; i++) {
            g1_step_3_vector[i] = my_zero;
            g1_step_4_vector[i] = my_zero;
        }
        for (int i = 0; i < D_N; i++) {
            g1_step_5_vector[i] = my_zero;
            g1_step_6_vector[i] = my_zero;
            g1_step_7_vector[i] = my_zero;
            g1_step_8_vector[i] = my_zero;
        }
        // STEP 3
        for (int eigenvector_index = 0; eigenvector_index < D_N_PLUS_ONE; eigenvector_index++) {
            complex<double> exponent(0.0,-w_N_plus_one[eigenvector_index]*times[t_index]);
            complex<double> bracket(g1_step_2_vector[eigenvector_index],0.0);
            g1_step_3_vector[eigenvector_index] = bracket * exp(exponent);
        }
        // STEP 4
        for (int y = 0; y < num_up_for_N_plus_1; y++) {
            for (int z = 0; z < num_down; z++) { // access |N+1>-basis, & case 2 for indexing
                 for (int eigenvector_index = 0; eigenvector_index < D_N_PLUS_ONE; eigenvector_index++) {
                     complex<double> weight(z_N_plus_one[(y*skip_l+z)*D_N_PLUS_ONE+eigenvector_index],0.0);
                     g1_step_4_vector[y*skip_l+z] = g1_step_4_vector[y*skip_l+z] + g1_step_3_vector[eigenvector_index] * weight;
                  }
            }
        }
        // STEP 5
        for (int w = 0; w < num_up_for_N_plus_1; w++) {
            for (int x = 0; x < num_down; x++) {
                complex<double> coefficient = g1_step_4_vector[w*skip_l+x]; // case 2 for indexing
                vector<int> output = apply_c_special(k_index_unprimed, up_spin_sets_in_bit_form_for_N_equals_5[w], coefficient, true, 0);
                for (int y = 0; y < num_up_for_N; y++) { // access |N>-basis
                    if (output == up_spin_sets_in_bit_form_for_N_equals_4[y]) {
                        int z = x; // down set does not change
                        g1_step_5_vector[z*skip_n+y] = g1_step_5_vector[z*skip_n+y] + coefficient; // case 1 for indexing
                    }
                }
            }
        }
        // STEP 6
        for (int eigenvector_index = 0; eigenvector_index < D_N; eigenvector_index++) {
            for (int element = 0; element < D_N; element++) {
                complex<double> value(z_N[element*D_N + eigenvector_index], 0.0);
                g1_step_6_vector[eigenvector_index] = g1_step_6_vector[eigenvector_index] + value * g1_step_5_vector[element];
            }
        }
        // STEP 7
        for (int eigenvector_index = 0; eigenvector_index < D_N; eigenvector_index++) {
            complex<double> exponent(0.0, w_N[eigenvector_index] * times[t_index] );
            complex<double> bracket = g1_step_6_vector[eigenvector_index];
            g1_step_7_vector[eigenvector_index] = bracket * exp(exponent);
        }
        // STEP 8
        for (int y = 0; y < num_up_for_N; y++) { // access |N>-basis, & case 1 for indexing
            for (int z = 0; z < num_down; z++) {
                for (int eigenvector_index = 0; eigenvector_index < D_N; eigenvector_index++) {
                    complex<double> weight(z_N[(z*skip_n+y)*D_N + eigenvector_index], 0.0);
                    g1_step_8_vector[z*skip_n+y] = g1_step_8_vector[z*skip_n+y] + g1_step_7_vector[eigenvector_index] * weight;
                }
            }
        }
        // STEP 9
        for (int element = 0; element < D_N; element++) {
            complex<double> value(ground_state[element], 0.0);
            g1[t_index] = g1[t_index] + value * g1_step_8_vector[element];
        }
        g1[t_index] = g1[t_index] * minus_i; // greater green's function (exactly)
    }
}
void green_part_2(int k_index_unprimed, int k_prime_index, double* ground_state, int N_up_original, int N_down_original) {
    
    cout << "obtaining up-spin sets for different values of N_up..." << endl;
    combinations_up.clear();
    int array_one[100];
    int n = V;
    int k = N_up_original; // N=4
    combinations_partial(array_one, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_4 = get_up_spin_sets_in_bit_form();
    combinations_up.clear();
    int array_two[100];
    n = V;
    k = N_up_original - 1; // N=3
    combinations_partial(array_two, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_3 = get_up_spin_sets_in_bit_form();
    
    long num_down = combinations_down.size(); // does not change throughout green's function calculation
    
    for (int i = 0; i < times.size(); i++) {
        g2[i] = my_zero;
    }
    
    for (int i = 0; i < D_N; i++) {
        step_1_vector[i] = 0.0;
    }

    cout << "starting step 1..." << endl;
    
    // STEP 1
    for (int eigenvector_index = 0; eigenvector_index < D_N; eigenvector_index++) {
        for (int element = 0; element < D_N; element++) {
            step_1_vector[eigenvector_index] = step_1_vector[eigenvector_index] + z_N[element*D_N + eigenvector_index] * ground_state[element];
        }
    }
    
    cout << "starting step 2..." << endl;
    // STEP 2
    for (int t_index = 0; t_index < times.size(); t_index++) {
        for (int i = 0; i < D_N; i++) {
            step_2_vector[i] = my_zero;
            step_3_vector[i] = my_zero;
            step_8_vector[i] = my_zero;
        }
        for (int i = 0; i < D_N_MINUS_ONE; i++) {
            step_4_vector[i] = my_zero;
            step_5_vector[i] = my_zero;
            step_6_vector[i] = my_zero;
            step_7_vector[i] = my_zero;
        }
        for (int eigenvector_index = 0; eigenvector_index < D_N; eigenvector_index++) {
            complex<double> exponent(0.0,-w_N[eigenvector_index]*times[t_index]);
            complex<double> bracket(step_1_vector[eigenvector_index],0.0);
            step_2_vector[eigenvector_index] = bracket * exp(exponent);
        }
        // STEP 3
        int skip_m;
        int skip_n;
        if (up_spin_sets_in_bit_form_for_N_equals_3.size() >= num_down) {
           skip_m = up_spin_sets_in_bit_form_for_N_equals_3.size();
        }
        else {
             skip_m = num_down;
        }
        if (up_spin_sets_in_bit_form_for_N_equals_4.size() >= num_down) {
           skip_n = up_spin_sets_in_bit_form_for_N_equals_4.size();
        }
        else {
             skip_n = num_down;
        }
        for (int y = 0; y < num_up_for_N; y++) {
            for (int z = 0; z < num_down; z++) { // access |N>-basis, & case 1
                for (int eigenvector_index = 0; eigenvector_index < D_N; eigenvector_index++) {
                    complex<double> weight(z_N[(z*skip_n+y)*D_N+eigenvector_index],0.0);
                    step_3_vector[z*skip_n+y] = step_3_vector[z*skip_n+y] + step_2_vector[eigenvector_index] * weight;
                }
            }
        }
        // STEP 4
        for (int w=0; w < num_up_for_N; w++) {
            for (int x=0; x < num_down; x++) {
                complex<double> coefficient = step_3_vector[x*skip_n+w];
                vector<int> output = apply_c_special(k_index_unprimed, up_spin_sets_in_bit_form_for_N_equals_4[w],coefficient,true,0);
                for (int y = 0; y < num_up_for_N_minus_1; y++) { // access |N-1>-basis
                    if (output == up_spin_sets_in_bit_form_for_N_equals_3[y]) {
                        int z = x; // down set does not change
                        step_4_vector[y*skip_m+z] = step_4_vector[y*skip_m+z] + coefficient; // case 2 for indexing
                    }
                }
            }
        }
        // STEP 5
        for (int eigenvector_index = 0; eigenvector_index < D_N_MINUS_ONE; eigenvector_index++) {
            for (int element = 0; element < D_N_MINUS_ONE; element++) {
                complex<double> value(z_N_minus_one[element*D_N_MINUS_ONE+eigenvector_index],0.0);
                step_5_vector[eigenvector_index] = step_5_vector[eigenvector_index] + value * step_4_vector[element];
            }
        }
        // STEP 6
        for (int eigenvector_index = 0; eigenvector_index < D_N_MINUS_ONE; eigenvector_index++) {
            complex<double> exponent(0.0,w_N_minus_one[eigenvector_index]*times[t_index]);
            complex<double> bracket = step_5_vector[eigenvector_index];
            step_6_vector[eigenvector_index] = bracket * exp(exponent);
        }
        // STEP 7
        for (int y = 0; y < num_up_for_N_minus_1; y++) { // access |N-1>-basis, & case 2 for indexing
            for (int z=0; z < num_down; z++) {
                for (int eigenvector_index = 0; eigenvector_index < D_N_MINUS_ONE; eigenvector_index++) {
                    complex<double> weight(z_N_minus_one[(y*skip_m+z)*D_N_MINUS_ONE+eigenvector_index],0.0);
                    step_7_vector[y*skip_m+z] = step_7_vector[y*skip_m+z] + step_6_vector[eigenvector_index] * weight;
                }
            }
        }
        // STEP 8
        for (int w = 0; w < num_up_for_N_minus_1; w++) {
            for (int x = 0; x < num_down; x++) {
                complex<double> coefficient = step_7_vector[w*skip_m+x]; // case 2 for indexing
                vector<int> output = apply_c_dagger_special(k_prime_index, up_spin_sets_in_bit_form_for_N_equals_3[w],coefficient, true, 0);
                for (int y = 0; y < num_up_for_N; y++) { // access |N>-basis
                    if (output == up_spin_sets_in_bit_form_for_N_equals_4[y]) {
                        int z = x; // down set does not change
                        step_8_vector[z*skip_n+y] = step_8_vector[z*skip_n+y] + coefficient; // case 1 for indexing
                    }
                }
            }
        }
        // STEP 9
        for (int element = 0; element < D_N; element++) {
            complex<double> value(ground_state[element],0.0);
            g2[t_index] = g2[t_index] + value * step_8_vector[element];
        }
        g2[t_index] = g2[t_index] * minus_i; // lesser green's function * (-1), keep the minus sign for the retarded green's function formulation
    }
}
void output_ground_state_green(int N_up_original, int N_down_original) {
    
    for (int k_index_unprimed = 0; k_index_unprimed <= V-1; k_index_unprimed++)
    {
        for (int k_prime_index = 0; k_prime_index <= V-1; k_prime_index++) {
        
            ofstream green_file_real;
            ofstream green_file_imag;
        
            string filename_real = "/Users/christinadaniel/Desktop/Christina_Desktop/data/t_interacting/ge_real" + to_string(k_index_unprimed) + "_" + to_string(k_prime_index) + ".txt";
            string filename_imag =  "/Users/christinadaniel/Desktop/Christina_Desktop/data/t_interacting/ge_imag" + to_string(k_index_unprimed) + "_" + to_string(k_prime_index) + ".txt";

            green_file_real.open(filename_real,ios::trunc);
            green_file_imag.open(filename_imag,ios::trunc);
        
            green_part_1(k_index_unprimed,k_prime_index,exact_ground_state,N_up_original,N_down_original);
            green_part_2(k_index_unprimed,k_prime_index,exact_ground_state,N_up_original,N_down_original);
        
            for (int i = 0; i < times.size(); i++) {
                green_file_real << times[i] << " " << real(g1[i]) + real(g2[i]) << endl;
                green_file_imag << times[i] << " " << imag(g1[i]) + imag(g2[i]) << endl;
            }
            green_file_real.close();
            green_file_imag.close();
        }
   }
    
}
void output_ground_state_lesser_and_greater(int N_up_original, int N_down_original) {
   
    for (int k_index_unprimed = 0; k_index_unprimed <= V-1; k_index_unprimed++) {
        for (int k_prime_index = 0; k_prime_index <= V-1; k_prime_index++) {
            
            ofstream lesser_file_real;
            ofstream greater_file_real;
            ofstream lesser_file_imag;
            ofstream greater_file_imag;
       
            string filename_lesser_real = "/Users/christinadaniel/Desktop/Christina_Desktop/data/t_interacting/lesser_real" + to_string(k_index_unprimed) + "_" + to_string(k_prime_index) + ".txt";
            string filename_lesser_imag = "/Users/christinadaniel/Desktop/Christina_Desktop/data/t_interacting/lesser_imag" + to_string(k_index_unprimed) + "_" + to_string(k_prime_index) + ".txt";
            string filename_greater_real = "/Users/christinadaniel/Desktop/Christina_Desktop/data/t_interacting/greater_real" + to_string(k_index_unprimed) + "_" + to_string(k_prime_index) + ".txt";
            string filename_greater_imag = "/Users/christinadaniel/Desktop/Christina_Desktop/data/t_interacting/greater_imag" + to_string(k_index_unprimed) + "_" + to_string(k_prime_index) + ".txt";

            lesser_file_real.open(filename_lesser_real,ios::trunc);
            greater_file_real.open(filename_greater_real,ios::trunc);
            lesser_file_imag.open(filename_lesser_imag,ios::trunc);
            greater_file_imag.open(filename_greater_imag,ios::trunc);

            green_part_1(k_index_unprimed,k_prime_index,exact_ground_state,N_up_original,N_down_original);
            green_part_2(k_index_unprimed,k_prime_index,exact_ground_state,N_up_original,N_down_original);

            for (int i = 0; i < times.size(); i++) {
                lesser_file_real << times[i] << " " << (-1) * real(g2[i]) << endl; // lesser green's function (exactly)
                greater_file_real << times[i] << " " << real(g1[i]) << endl; // greater green's function (exactly)
                lesser_file_imag << times[i] << " " << (-1) * imag(g2[i]) << endl; // lesser green's function (exactly)
                greater_file_imag << times[i] << " " << imag(g1[i]) << endl; // greater green's function (exactly)
                
            }
            lesser_file_real.close();
            greater_file_real.close();
            lesser_file_imag.close();
            greater_file_imag.close();
        }
        
    }

}



void fill_matrices_part_1(int k_index_unprimed, int k_prime_index, int N_up_original, int N_down_original) {
    
    combinations_up.clear();
    int array_one[100];
    int n = V;
    int k = N_up_original; // N=4
    combinations_partial(array_one, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_4 = get_up_spin_sets_in_bit_form();
    combinations_up.clear();
    int array_two[100];
    n = V;
    k = N_up_original - 1; // N=3
    combinations_partial(array_two, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_3 = get_up_spin_sets_in_bit_form();
    combinations_up.clear();
    int array_three[100];
    n = V;
    k = N_up_original + 1; // N=5
    combinations_partial(array_three, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_5 = get_up_spin_sets_in_bit_form();
    
    long num_down = combinations_down.size(); // does not change throughout green's function calculation
       
    cout << "initializing matrix elements" << endl;
    for (int n = 0; n < D_N; n++) {
        for (int l = 0; l < D_N_PLUS_ONE; l++) {
            matrix_n_c_l[n*D_N_PLUS_ONE + l] = 0.0;
            matrix_l_c_dagger_n[l*D_N + n] = 0.0;
        }
    }
    
    cout << "allocating c maps" << endl;
    int c_key;
    for (int w = 0; w < num_up_for_N_plus_1; w++) { // access unique up-spin sets in |N+1>-basis
        complex<double> coefficient = my_one;
        vector<int> output = apply_c_special(k_index_unprimed, up_spin_sets_in_bit_form_for_N_equals_5[w], coefficient, true, 0);
        c_key = w;
        map_for_c_signs_PART_1.insert(make_pair(c_key,coefficient));
        for (int w_out = 0; w_out < num_up_for_N; w_out++) {
            if (output == up_spin_sets_in_bit_form_for_N_equals_4[w_out]) {
                map_for_c_on_ket_to_index_PART_1.insert(make_pair(c_key,w_out));
            }
        }
    }

    cout << "allocating c-dagger maps" << endl;
    int c_dagger_key;
    for (int w = 0; w < num_up_for_N; w++) { // access unique up-spin sets in |N>-basis
        complex<double> coefficient = my_one;
        vector<int> output = apply_c_dagger_special(k_prime_index, up_spin_sets_in_bit_form_for_N_equals_4[w], coefficient, true, 0);
        c_dagger_key = w;
        map_for_c_dagger_signs_PART_1.insert(make_pair(c_dagger_key,coefficient));
        for (int w_out = 0; w_out < num_up_for_N_plus_1; w_out++) {
            if (output == up_spin_sets_in_bit_form_for_N_equals_5[w_out]) {
                map_for_c_dagger_on_ket_to_index_PART_1.insert(make_pair(c_dagger_key,w_out));
            }
        }
    }
    
    int up_output_index;
    complex<double> coeff;
    
    cout << "allocating c matrix" << endl;
    // allocate <n|c|l>
    int skip_l;
    int skip_n;
    if (up_spin_sets_in_bit_form_for_N_equals_5.size() >= num_down) {
       skip_l = up_spin_sets_in_bit_form_for_N_equals_5.size();
    }
    else {
         skip_l = num_down;
    }
    if (up_spin_sets_in_bit_form_for_N_equals_4.size() >= num_down) {
       skip_n = up_spin_sets_in_bit_form_for_N_equals_4.size();
    }
    else {
         skip_n = num_down;
    }
    for (int w = 0; w < num_up_for_N_plus_1; w++) { // access unique up-spin sets in |N+1>-basis
        if ( map_for_c_signs_PART_1.find(w)->second != my_zero ) {
            for (int n = 0; n < D_N; n++) { // access energy eigenstate |n> in |N>-basis
                for (int l = 0; l < D_N_PLUS_ONE; l++) { // access energy eigenstate |l> in |N+1>-basis
                    for (int x = 0; x < num_down; x++) { // access unique down-spin sets
                        up_output_index = map_for_c_on_ket_to_index_PART_1.find(w)->second;
                        int index_n;
                        if (up_spin_sets_in_bit_form_for_N_equals_4.size() >= num_down) {
                           index_n = x*skip_n+up_output_index;
                        }
                        else {
                             index_n = up_output_index*skip_n+x;
                        }
                        int index_l;
                        if (up_spin_sets_in_bit_form_for_N_equals_5.size() >= num_down) {
                           index_l = x*skip_l + w;
                        }
                        else {
                             index_l = w*skip_l + x;
                        }
                        complex<double> n_eigenstate_coeff(z_N[index_n*D_N + n],0.0);
                        complex<double> l_eigenstate_coeff(z_N_plus_one[index_l*D_N_PLUS_ONE + l],0.0);
                        coeff = n_eigenstate_coeff * l_eigenstate_coeff * map_for_c_signs_PART_1.find(w)->second;
                        matrix_n_c_l[n*D_N_PLUS_ONE + l] = matrix_n_c_l[n*D_N_PLUS_ONE + l] + real(coeff);
                    }
                }
            }
        }
    }
    cout << "allocating c-dagger matrix" << endl;
    // allocate <l|c-dagger|n>
    for (int w = 0; w < num_up_for_N; w++) {
        if ( map_for_c_dagger_signs_PART_1.find(w)->second != my_zero ) {
            for (int n = 0; n < D_N; n++) { // access energy eigenstate |n> in |N>-basis
                for (int l = 0; l < D_N_PLUS_ONE; l++) { // access energy eigenstate |l> in |N+1>-basis
                    for (int x = 0; x < num_down; x++) {
                        up_output_index = map_for_c_dagger_on_ket_to_index_PART_1.find(w)->second;
                        int index_n;
                        if (up_spin_sets_in_bit_form_for_N_equals_4.size() >= num_down) {
                           index_n = x*skip_n+w;
                        }
                        else {
                             index_n = w*skip_n+x;
                        }
                        int index_l;
                        if (up_spin_sets_in_bit_form_for_N_equals_5.size() >= num_down) {
                           index_l = x*skip_l+up_output_index;
                        }
                        else {
                             index_l = up_output_index*skip_l+x;
                        }
                        complex<double> n_eigenstate_coeff(z_N[index_n*D_N + n],0.0);
                        complex<double> l_eigenstate_coeff(z_N_plus_one[index_l*D_N_PLUS_ONE + l],0.0);
                        coeff = n_eigenstate_coeff * l_eigenstate_coeff * map_for_c_dagger_signs_PART_1.find(w)->second;
                        matrix_l_c_dagger_n[l*D_N + n] = matrix_l_c_dagger_n[l*D_N + n] + real(coeff); // lth element/row of nth eigenvector/col
                    }
                }
            }
        }
    }
    cout << "matrices for temperature-dependent retarded green's function (part 1) are filled." << endl;
}
void fill_matrices_part_2(int k_index_unprimed, int k_prime_index, int N_up_original, int N_down_original) {
    
    combinations_up.clear();
    int array_one[100];
    int n = V;
    int k = N_up_original; // N=4
    combinations_partial(array_one, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_4 = get_up_spin_sets_in_bit_form();
    combinations_up.clear();
    int array_two[100];
    n = V;
    k = N_up_original - 1; // N=3
    combinations_partial(array_two, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_3 = get_up_spin_sets_in_bit_form();
    combinations_up.clear();
    int array_three[100];
    n = V;
    k = N_up_original + 1; // N=5
    combinations_partial(array_three, 1, n, 1, k, true);
    vector<vector<int> > up_spin_sets_in_bit_form_for_N_equals_5 = get_up_spin_sets_in_bit_form();
    
    long num_down = combinations_down.size(); // does not change throughout green's function calculation
    
    cout << "initializing matrix elements" << endl;
    for (int n=0; n < D_N; n++) {
        for (int m = 0; m < D_N_MINUS_ONE; m++) {
            matrix_m_c_n[m*D_N + n] = 0.0;
            matrix_n_c_dagger_m[n*D_N_MINUS_ONE + m] = 0.0;
        }
    }
    
    cout << "allocating c maps" << endl;
    int c_key;
    for (int w = 0; w < num_up_for_N; w++) { // access unique up-spin sets in |N>-basis
        complex<double> coefficient = my_one;
        vector<int> output = apply_c_special(k_index_unprimed, up_spin_sets_in_bit_form_for_N_equals_4[w], coefficient, true, 0);
        c_key = w;
        map_for_c_signs.insert(make_pair(c_key,coefficient));
        for (int w_out = 0; w_out < num_up_for_N_minus_1; w_out++) {
            if (output == up_spin_sets_in_bit_form_for_N_equals_3[w_out]) {
                map_for_c_on_ket_to_index.insert(make_pair(c_key,w_out));
            }
        }
    }
    cout << "allocating c-dagger maps" << endl;
    int c_dagger_key;
    for (int w = 0; w < num_up_for_N_minus_1; w++) { // access unique up-spin sets in |N-1>-basis
        complex<double> coefficient = my_one;
        vector<int> output = apply_c_dagger_special(k_prime_index, up_spin_sets_in_bit_form_for_N_equals_3[w], coefficient, true, 0);
        c_dagger_key = w;
        map_for_c_dagger_signs.insert(make_pair(c_dagger_key,coefficient));
        for (int w_out = 0; w_out < num_up_for_N; w_out++) {
            if (output == up_spin_sets_in_bit_form_for_N_equals_4[w_out]) {
                map_for_c_dagger_on_ket_to_index.insert(make_pair(c_dagger_key,w_out));
            }
        }
    }
    
    int up_output_index;
    complex<double> coeff;
    
    cout << "allocating c matrix" << endl;
    // allocate <m|c|n>
    int skip_n;
    int skip_m;
    if (up_spin_sets_in_bit_form_for_N_equals_4.size() >= num_down) {
       skip_n = up_spin_sets_in_bit_form_for_N_equals_4.size();
    }
    else {
         skip_n = num_down;
    }
    if (up_spin_sets_in_bit_form_for_N_equals_3.size() >= num_down) {
       skip_m = up_spin_sets_in_bit_form_for_N_equals_3.size();
    }
    else {
         skip_m = num_down;
    }
    for (int w = 0; w < num_up_for_N; w++) { // access unique up-spin sets in |N>-basis
        if ( map_for_c_signs.find(w)->second != my_zero ) {
            for (int n = 0; n < D_N; n++) { // access energy eigenstate |n> in |N>-basis
                for (int m = 0; m < D_N_MINUS_ONE; m++) { // access energy eigenstate |m> in |N-1>-basis
                    for (int x = 0; x < num_down; x++) { // access unique down-spin sets
                        up_output_index = map_for_c_on_ket_to_index.find(w)->second;
                        int index_n;
                        if (up_spin_sets_in_bit_form_for_N_equals_4.size() >= num_down) {
                           index_n = x*skip_n+w;
                        }
                        else {
                             index_n = w*skip_n+x;
                        }
                        int index_m;
                        if (up_spin_sets_in_bit_form_for_N_equals_3.size() >= num_down) {
                           index_m = x*skip_m+up_output_index;
                        }
                        else {
                             index_m = up_output_index*skip_m+x;
                        }
                        complex<double> n_eigenstate_coeff(z_N[index_n*D_N + n],0.0); // (x*skip+w)th element of nth eigenvector
                        complex<double> m_eigenstate_coeff(z_N_minus_one[index_m*D_N_MINUS_ONE + m],0.0); // (up_output_index*skip+x)th element of mth eigenvector
                        coeff = n_eigenstate_coeff * m_eigenstate_coeff * map_for_c_signs.find(w)->second;
                        matrix_m_c_n[m*D_N + n] = matrix_m_c_n[m*D_N + n] + real(coeff); // mth row of nth column
                    }
                }
            }
        }
    }
    cout << "allocating c-dagger matrix" << endl;
    // allocate <n|c-dagger|m>
    for (int w = 0; w < num_up_for_N_minus_1; w++) {
        if ( map_for_c_dagger_signs.find(w)->second != my_zero ) {
            for (int n = 0; n < D_N; n++) { // access energy eigenstate |n> in |N>-basis
                for (int m = 0; m < D_N_MINUS_ONE; m++) { // access energy eigenstate |m> in |N-1>-basis
                    for (int x = 0; x < num_down; x++) {
                        up_output_index = map_for_c_dagger_on_ket_to_index.find(w)->second;
                        int index_n;
                        if (up_spin_sets_in_bit_form_for_N_equals_4.size() >= num_down) {
                           index_n = x*skip_n+up_output_index;
                        }
                        else {
                             index_n = up_output_index*skip_n+x;
                        }
                        int index_m;
                        if (up_spin_sets_in_bit_form_for_N_equals_3.size() >= num_down) {
                           index_m = x*skip_m+w;
                        }
                        else {
                             index_m = w*skip_m+x;
                        }
                        complex<double> n_eigenstate_coeff(z_N[index_n*D_N + n],0.0); // case 1 for indexing
                        complex<double> m_eigenstate_coeff(z_N_minus_one[index_m*D_N_MINUS_ONE + m],0.0); // case 2 for indexing
                        coeff = n_eigenstate_coeff * m_eigenstate_coeff * map_for_c_dagger_signs.find(w)->second;
                        matrix_n_c_dagger_m[n*D_N_MINUS_ONE + m] = matrix_n_c_dagger_m[n*D_N_MINUS_ONE + m] + real(coeff); // nth row of mth column
                    }
                }
            }
        }
    }
    cout << "matrices for temperature-dependent retarded green's function (part 2) are filled." << endl;
}
void clear_part_1_maps() {
    // part 1
    map_for_c_signs_PART_1.clear();
    map_for_c_on_ket_to_index_PART_1.clear();
    map_for_c_dagger_signs_PART_1.clear();
    map_for_c_dagger_on_ket_to_index_PART_1.clear();
}
void clear_part_2_maps() {
    // part 2
    map_for_c_signs.clear();
    map_for_c_on_ket_to_index.clear();
    map_for_c_dagger_signs.clear();
    map_for_c_dagger_on_ket_to_index.clear();
}
void n_minus_one(int k_index_unprimed, int k_prime_index, int N_up_original, int N_down_original) {

    int n = GROUND_STATE_INDEX;
    for (int m = 0; m < D_N_MINUS_ONE; m++) {
        long double element_1 = matrix_m_c_n[m*D_N + n];
        long double element_2 = matrix_n_c_dagger_m[n*D_N_MINUS_ONE + m];
        omegas_n_minus_one.push_back(w_N[n] - w_N_minus_one[m]);
        matrix_elements_n_minus_one.push_back(element_1*element_2);
    }
}
void n_plus_one(int k_index_unprimed, int k_prime_index, int N_up_original, int N_down_original) {

    int n = GROUND_STATE_INDEX;
    for (int l = 0; l < D_N_PLUS_ONE; l++) {
        long double element_1 = matrix_n_c_l[n*D_N_PLUS_ONE + l];
        long double element_2 = matrix_l_c_dagger_n[l*D_N + n];
        omegas_n_plus_one.push_back(w_N_plus_one[l] - w_N[n]);
        matrix_elements_n_plus_one.push_back(element_1*element_2);
    }
}
void shift_eigenvalues(int N_up_original, int N_down_original) {
    double mu = V_eff * 0.5; // chemical potential
    double N_original = N_up_original + N_down_original;
    for (int i = 0; i < D_N; i++) {
        w_N[i] = w_N[i] - mu*N_original;
    }
    for (int i = 0; i < D_N_MINUS_ONE; i++) {
        w_N_minus_one[i] = w_N_minus_one[i] - mu*(N_original-1);
    }
    for (int i = 0; i < D_N_PLUS_ONE; i++) {
        w_N_plus_one[i] = w_N_plus_one[i] - mu*(N_original+1);
    }
}
Symbolic construct_polynomial(int N_up_original, int N_down_original, double K_value) {
    
    Symbolic z("z");
    
    fill_matrices_part_1(0, 0, N_up_original, N_down_original);
    fill_matrices_part_2(0, 0, N_up_original, N_down_original);
    
    n_minus_one(0, 0, N_up_original, N_down_original);
    n_plus_one(0, 0, N_up_original, N_down_original);
    
    /* Form the product over alpha  */
    Symbolic product_over_alpha = 1;
    for (int a = 0; a < omegas_n_minus_one.size(); a++) {
        product_over_alpha = product_over_alpha * (z - (double) omegas_n_minus_one[a] );
    }
    /* Form the product over beta  */
    Symbolic product_over_beta = 1;
    for (int b = 0; b < omegas_n_plus_one.size(); b++) {
        product_over_beta = product_over_beta * (z - (double) omegas_n_plus_one[b] );
    }
    
    clear_part_1_maps();
    clear_part_2_maps();
    
    omegas_n_plus_one.clear();
    omegas_n_minus_one.clear();
    matrix_elements_n_plus_one.clear();
    matrix_elements_n_minus_one.clear();

    /* Calculate Denominator */
    Symbolic sum_over_ija = 0;
    Symbolic sum_over_ijb = 0;
    for (int i_index = 0; i_index <= V-1; i_index++) {
        for (int j_index = 0; j_index <= V-1; j_index++) {
    
        fill_matrices_part_1(i_index, j_index, N_up_original, N_down_original);
        fill_matrices_part_2(i_index, j_index, N_up_original, N_down_original);
            
        n_minus_one(i_index, j_index, N_up_original, N_down_original);
        n_plus_one(i_index, j_index, N_up_original, N_down_original);
       
        /* 4-Site index adjustment for cosine function */
        double cosine_value = 0;
        int adjusted_i_index = i_index;
        int adjusted_j_index = j_index;
        if (i_index == V-1) {
            adjusted_i_index = -1;
        }
        if (j_index == V-1) {
            adjusted_j_index = -1;
        }
        cosine_value = cos(K_value * ( adjusted_i_index - adjusted_j_index  ) );
            
        for (int a = 0; a < D_N_MINUS_ONE; a++) {
            Symbolic product_over_alpha_not_a = 1;
            for (int my_alpha = 0; my_alpha < D_N_MINUS_ONE; my_alpha++) {
                if (my_alpha != a) {
                    product_over_alpha_not_a = product_over_alpha_not_a * (z - (double) omegas_n_minus_one[a] );
                }
            }
            sum_over_ija = sum_over_ija + cosine_value * ( (double) matrix_elements_n_minus_one[a] ) * product_over_alpha_not_a;
        }
        for (int b = 0; b < D_N_PLUS_ONE; b++) {
            Symbolic product_over_beta_not_b = 1;
            for (int my_beta = 0; my_beta < D_N_PLUS_ONE; my_beta++) {
                if (my_beta != b) {
                    product_over_beta_not_b = product_over_beta_not_b * (z - (double) omegas_n_plus_one[b] );
                }
                
            }
            sum_over_ijb = sum_over_ijb + cosine_value * ( (double) matrix_elements_n_plus_one[b] ) * product_over_beta_not_b;
        }
            
        clear_part_1_maps();
        clear_part_2_maps();
            
        omegas_n_plus_one.clear();
        omegas_n_minus_one.clear();
        matrix_elements_n_plus_one.clear();
        matrix_elements_n_minus_one.clear();
           
        }
   }
    
   /* Form Denominator */
   Symbolic denominator = product_over_alpha * sum_over_ijb + product_over_beta * sum_over_ija;
   
   /* Print Denominator */
   cout << "Denominator:" << endl;
   cout.precision(dbl::max_digits10);
   cout << denominator << endl;
    
   /* Print Coefficients
   int degree = 47;
   for (int power = degree; power >= 0; power = power - 1) {
       cout << denominator.coeff(z,power) << "," << endl;
   } */

   /* Return Denominator */
   return denominator;
}



/* Solve linear or quadratic equation
 */
static void quadratic( const int n, const double a[], complex<double> res[] )
   {
   double r;

   if( n == 2 )
      {
      if( a[ 1 ] == 0 )
         {
         r = - a[ 2 ] / a [ 0 ];
         if( r < 0 )
            {
            res[ 1 ] = complex<double> ( 0, sqrt( - r ) );
            res[ 2 ] = complex<double> ( 0, -res[ 1 ].imag() );
            }
         else
            {
            res[ 1 ] = complex<double> ( sqrt( r ), 0 );
            res[ 2 ] = complex<double> ( -res[ 1 ].real(), 0 );
            }
         }
      else
         {
         r = 1 - 4 * a[ 0 ] * a[ 2 ] / ( a[ 1 ] * a[ 1 ] );
         if( r < 0 )
            {
            res[ 1 ] = complex<double> ( -a[ 1 ] / ( 2 * a[ 0 ] ), a[ 1 ] * sqrt( -r ) / ( 2 * a[ 0 ] ) );
            res[ 2 ] = complex<double> ( res[ 1 ].real(), -res[ 1 ].imag() );
            }
          else
             {
             res[ 1 ] = complex<double> ( ( -1 -sqrt( r ) ) * a[ 1 ] / ( 2 * a[ 0 ] ), 0 );
             res[ 2 ] = complex<double> ( a[ 2 ] / ( a[ 0 ] * res[ 1 ].real() ), 0 );
             }
         }
      }
   else
      if( n == 1 )
        res[ 1 ] = complex<double> ( -a[ 1 ] / a[ 0 ], 0 );
   }

/* Performed function evaluation. Horners algorithm.
 */
static double feval( const int n, const double a[], const complex<double> z, complex<double> *fz )
   {
   int i;
   double p, q, r, s, t;

   p = -2.0 * z.real();
   q = z.real() * z.real() + z.imag() * z.imag();
   s = 0; r = a[ 0 ];
   for( i = 1; i < n; i++ )
      {
      t = a[ i ] - p * r - q * s;
      s = r;
      r = t;
      }
   *fz = complex<double>( a[ n ] + z.real() * r - q * s, z.imag() * r );
   
   return fz->real() * fz->real() + fz->imag() * fz->imag();
   }


static double startpoint( const int n, const double a[] )
   {
   int i;
   double r, u, min;

   /* Determine starting point */
   r = log( fabs( a[ n ] ) );
   min = exp( ( r - log( fabs( a[ 0 ] ) ) ) / n );
   for( i = 1; i < n; i++ )
      if( a[ i ] != 0 )
         {
         u = exp( ( r - log( fabs( a[ i ] ) ) ) / ( n - i ) );
         if( u < min )
            min = u;
         }

   return min;
   }

/* Calculate a upper bound for the rounding errors performed in a
   polynomial at a complex point.
   ( Adam's test )
 */
static double upperbound( const int n, const double a[], const complex<double> z )
   {
   int i;
   double p, q, r, s, t, u, e;

   p = - 2.0 * z.real();
   q = z.real() * z.real() + z.imag() * z.imag();
   u = sqrt( q );
   s = 0.0; r = a[ 0 ]; e = fabs( r ) * ( 3.5 / 4.5 );
   for( i = 1; i < n; i++ )
      {
      t = a[ i ] - p * r - q * s;
      s = r;
      r = t;
      e = u * e + fabs( t );
      }
   t = a[ n ] + z.real() * r - q * s;
   e = u * e + fabs( t );
   e = ( 9.0 * e - 7.0 * ( fabs( t ) + fabs( r ) * u ) +
       fabs( z.real() ) * fabs( r ) * 2.0 ) * pow( FLT_RADIX, -DBL_MANT_DIG+1);

   return e * e;
   }

static void alterdirection( complex<double> *dz, const double m )
   {
   double x, y;

   x = ( dz->real() * 0.6 - dz->imag() * 0.8 ) * m;
   y = ( dz->real() * 0.8 + dz->imag() * 0.6 ) * m;
   *dz = complex<double> ( x, y );
   }


/* Real root forward deflation.
 */
static int realdeflation( const int n, double a[], const double x )
   {
   int i;
   double r;

   for( r = 0, i = 0; i < n; i++ )
      a[ i ] = r = r * x + a[ i ];
   return n - 1;
   }

/* Complex root forward deflation.
 */
static int complexdeflation( const int n, double a[], const complex<double> z )
   {
   int i;
   double r, u;

   r = -2.0 * z.real();
   u = z.real() * z.real() + z.imag() * z.imag();
   a[ 1 ] -= r * a[ 0 ];
   for( i = 2; i < n - 1; i++ )
      a[ i ] = a[ i ] - r * a[ i - 1 ] - u * a[ i - 2 ];

   return n - 2;
   }

// Find all root of a polynomial of n degree with real coeeficient using the modified Newton by Madsen
//
int newton_real( int n, const double coeff[], complex<double> res[] )
   {
   int i, itercnt;
   int stage1, div2;
   int err;
   double *a, *a1;
   double u, r, r0, eps;
   double f, f0, ff, f2, fw;
   complex<double> z, z0, dz, fz, fwz, wz, fz0, fz1;

   a = new double [ n+1 ];
    for( i = 0; i <= n; i++ )
        a[ i ] = coeff[ i ];

   err = 0;
   for( ; a[ n ] == 0.0; n-- )
      res[ n ] = complex<double> (0);

   a1 = new double [ n ];
   while( n > 2 )
      {
      /* Calculate coefficients of f'(x) */
      for( i = 0; i < n; i++ )
         a1[ i ] = a[ i ] * ( n - i );

      u = startpoint( n, a );
      z0 = complex<double> (0);
      f0 = ff = 2.0 * a[ n ] * a[ n ];
      fz0 = complex<double> ( a[ n - 1 ] );
      z = complex<double> ( a[ n - 1 ] == 0.0 ? 1 : -a[ n ] / a[ n - 1 ], 0 );
      z = complex<double> ( z.real() / (double)fabs( z.real() ) * u * 0.5, 0 );
      dz = z;
      f = feval( n, a, z, &fz );
      r0 = 2.5 * u;
      r = sqrt( dz.real() * dz.real() + dz.imag() * dz.imag() );
      eps = 4 * n * n * f0 * pow( FLT_RADIX, -DBL_MANT_DIG * 2 );

      // Start iteration
      for( stage1 = 1, itercnt = 0; ( z.real() + dz.real() != z.real() || z.imag() + dz.imag() != z.imag() )
                        &&  f > eps && itercnt < MAXITER; itercnt++ )
         {  /* Iterativ loop */
         u = feval( n - 1, a1, z, &fz1 );
         if( u == 0.0 )  /* True saddelpoint */
            alterdirection( &dz, 5.0 );
         else
            {
            dz = complex<double> ( ( fz.real() * fz1.real() + fz.imag() * fz1.imag() ) / u, ( fz.imag() * fz1.real() - fz.real() * fz1.imag() ) / u );

            /* Which stage are we on */
            fwz = fz0 - fz1;
            wz = z0 - z;
            f2 = ( fwz.real() * fwz.real() + fwz.imag() * fwz.imag() ) / ( wz.real() * wz.real() + wz.imag() * wz.imag() );
            stage1 = f2/u > u/f/4 || f != ff;
            r = sqrt( dz.real() * dz.real() + dz.imag() * dz.imag() );
            if( r > r0 )
               alterdirection( &dz, r0 / r );
            r0 = r * 5.0;
            }

         z0 = z;
         f0 = f; fz0 = fz;

iter2:
         z = z0 - dz;
         ff = f = feval( n, a, z, &fz );
         if( stage1 )
            {
            wz = z;
            for( i = 1, div2 = f > f0; i <= n; i++ )
               {
               if( div2 )
                  {
                  dz *= complex<double> (0.5);
                  wz = z0 - dz;
                  }
               else
                  {
                  wz -= dz;
                  }
               fw = feval( n, a, wz, &fwz );
               if( fw >= f )
                  break;
               f = fw;
               fz = fwz;
               z = wz;
               if( div2 && i == 2 )
                  {
                  alterdirection( &dz, 0.5 );
                  z = z0 - dz;
                  f = feval( n, a, z, &fz );
                  break;
                  }
               }
            }
         else
            {
            /* calculate the upper bound of erros using Adam's test */
            eps = upperbound( n, a, z );
            }

         if( r < sqrt( z.real() * z.real() + z.imag() * z.imag() ) * pow( FLT_RADIX, -DBL_MANT_DIG/2 ) && f >= f0 )
            {
            /* Domain rounding errors */
            z = z0;
            alterdirection( &dz, 0.5 );
            if( z + dz != z )
               goto iter2;
            }
         }

      if( itercnt >= MAXITER )
         err--;

      z0 = complex<double> (z.real(), 0.0 );
      if( feval( n, a, z0, &fz ) <= f )
         {
         /* Real root */
         res[ n ] = complex<double> ( z.real(), 0 );
         n = realdeflation( n, a, z.real() );
         }
      else
         {
         /* Complex root */
         res[ n ] = z;
         res[ n - 1 ] = complex<double>( z.real(), -z.imag() );
         n = complexdeflation( n, a, z );
         }
      }

   quadratic( n, a, res );

   delete [] a1;
   delete [] a;

   return( err );
   }


void output_values(int N_up_original, int N_down_original) {

     for (int k_index_unprimed = 0; k_index_unprimed <= V-1; k_index_unprimed++)
     {
         for (int k_prime_index = 0; k_prime_index <= V-1; k_prime_index++) {

         fill_matrices_part_1(k_index_unprimed, k_prime_index, N_up_original, N_down_original);
         fill_matrices_part_2(k_index_unprimed, k_prime_index, N_up_original, N_down_original);

         ofstream n_plus_one_file;
         ofstream n_minus_one_file;

         string filename_n_plus_one = "/Users/christinadaniel/Desktop/Christina_Desktop/data/w_interacting/n_plus_one" + to_string(k_index_unprimed) + "_" + to_string(k_prime_index) + ".txt";
         string filename_n_minus_one = "/Users/christinadaniel/Desktop/Christina_Desktop/data/w_interacting/n_minus_one" + to_string(k_index_unprimed) + "_" + to_string(k_prime_index) + ".txt";

         n_plus_one_file.open(filename_n_plus_one,ios::trunc);
         n_minus_one_file.open(filename_n_minus_one,ios::trunc);

         n_minus_one(k_index_unprimed, k_prime_index, N_up_original, N_down_original);
         n_plus_one(k_index_unprimed, k_prime_index, N_up_original, N_down_original);

         for (int i = 0; i < omegas_n_plus_one.size(); i++) {
             n_plus_one_file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << omegas_n_plus_one[i] << " " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << real(matrix_elements_n_plus_one[i]) << endl;
         }

         for (int i = 0; i < omegas_n_minus_one.size(); i++) {
             n_minus_one_file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << omegas_n_minus_one[i] << " " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << real(matrix_elements_n_minus_one[i]) << endl;
         }

         n_plus_one_file.close();
         n_minus_one_file.close();

         // IMPORTANT: CLEAR IF ITERATING OVER DIFFERENT INDICES
         clear_part_1_maps();
         clear_part_2_maps();

         omegas_n_plus_one.clear();
         omegas_n_minus_one.clear();
         matrix_elements_n_plus_one.clear();
         matrix_elements_n_minus_one.clear();

         }
    }
}




int main (int argc, char *argv[]) {

    z_N = new double[array_size_for_N];
    z_N_minus_one = new double[array_size_for_N_minus_1];
    z_N_plus_one = new double[array_size_for_N_plus_1];
    
    matrix_m_c_n = new double[D_N_MINUS_ONE*D_N];
    matrix_n_c_dagger_m = new double[D_N*D_N_MINUS_ONE];
    matrix_n_c_l = new double[D_N*D_N_PLUS_ONE];
    matrix_l_c_dagger_n = new double[D_N_PLUS_ONE*D_N];
    
    hamiltonian_N = new double[array_size_for_N];
    hamiltonian_N_minus_one = new double[array_size_for_N_minus_1];
    hamiltonian_N_plus_one = new double[array_size_for_N_plus_1];
    
    ground_state_N_minus_one = new double[D_N_MINUS_ONE];
    ground_state_N_plus_one = new double[D_N_PLUS_ONE];
    
    kinetic.resize(array_size_for_N);
    potential.resize(array_size_for_N);
    fill(kinetic.begin(),kinetic.end(),my_zero);
    fill(potential.begin(),potential.end(),my_zero);
    
    int N_up_original = 2;
    int N_down_original = 2;
    int v[100];
    int n = V;
    int k = N_up_original;
    combinations_partial(v, 1, n, 1, k, true);
    n = V;
    k = N_down_original;
    int w[100];
    combinations_partial(w, 1, n, 1, k, false);
    fill_maps(N_up_original, N_down_original);
    convert_from_ket_to_index_of_part(N_up_original, N_down_original);
    int D = D_N;
    int N_up = N_up_original;
    int N_down = N_down_original;
    matrix_for_T(N_up_original, N_down_original);
    matrix_for_U(N_up_original, N_down_original);
    add_matrices(D);
    convert(D,hamiltonian_N);
    diagonalize(D, exact_ground_state, hamiltonian_N, z_N, w_N, ifail_N);
    kinetic.resize(array_size_for_N_minus_1);
    potential.resize(array_size_for_N_minus_1);
    fill(kinetic.begin(),kinetic.end(),my_zero);
    fill(potential.begin(),potential.end(),my_zero);
    clear_all_maps();
    delete[] hamiltonian_N;

    if (print==true) {
       cout << "Basis states for N electrons (comma delimited):" << endl;
       vector<vector<int> > up_spin_sets_in_bit_form = get_up_spin_sets_in_bit_form();
       vector<vector<int> > down_spin_sets_in_bit_form = get_down_spin_sets_in_bit_form();
       int num_down = down_spin_sets_in_bit_form.size();
       int skip_n;
       if (up_spin_sets_in_bit_form.size() >= num_down) { // case 1 for indexing; up-spin sets vary faster
             skip_n = up_spin_sets_in_bit_form.size();
          for (int x = 0; x < num_down; x++) {
              for (int w = 0; w < up_spin_sets_in_bit_form.size(); w++) {
                  for (int up_index = 0; up_index < V; up_index++) {
                      cout << up_spin_sets_in_bit_form[w][up_index] << " ";
                  }
                  cout << "| ";
                  for (int down_index = 0; down_index < V; down_index++) {
                      cout << down_spin_sets_in_bit_form[x][down_index] << " ";
                  }
                  cout << ", ";
              }
          }
       }
       else { // case 2 for indexing; down-spin sets vary faster
            skip_n = num_down;
            for (int w = 0; w < up_spin_sets_in_bit_form.size(); w++) {
                for (int x = 0; x < num_down; x++) {
                    for (int up_index = 0; up_index < V; up_index++) {
                        cout << up_spin_sets_in_bit_form[w][up_index] << " ";
                    }
                    cout << "| ";
                    for (int down_index = 0; down_index < V; down_index++) {
                        cout << down_spin_sets_in_bit_form[x][down_index] << " ";
                    }
                    cout << ", ";
                }
            }
       }
       cout << endl;
       cout << "Eigenvalues for N electrons:" << endl;
       for (int i = 0; i < D; i++) {
           cout << w_N[i] << endl;
       }
       cout << "Eigenvectors (one per row) for N electrons:" << endl;
       for (int j = 0; j < D; j++) {
           for (int i = 0; i < D; i++) {
               cout << z_N[i*D+j] << " ";
           }
           cout << endl;
       }
    }
    
    N_up = N_up_original - 1;
    N_down = N_down_original;
    combinations_up.clear();
    combinations_down.clear();
    int x[100];
    n = V;
    k = N_up;
    combinations_partial(x, 1, n, 1, k, true);
    n = V;
    k = N_down;
    int y[100];
    combinations_partial(y, 1, n, 1, k, false);
    fill_maps(N_up, N_down);
    convert_from_ket_to_index_of_part(N_up, N_down);
    int D2 = D_N_MINUS_ONE;
    matrix_for_T(N_up, N_down);
    matrix_for_U(N_up, N_down);
    add_matrices(D2);
    convert(D2,hamiltonian_N_minus_one);
    diagonalize(D2, ground_state_N_minus_one, hamiltonian_N_minus_one, z_N_minus_one, w_N_minus_one, ifail_N_minus_one);
    kinetic.resize(array_size_for_N_plus_1);
    potential.resize(array_size_for_N_plus_1);
    fill(kinetic.begin(),kinetic.end(),my_zero);
    fill(potential.begin(),potential.end(),my_zero);
    clear_all_maps();
    delete[] hamiltonian_N_minus_one;
    delete[] ground_state_N_minus_one;

    if (print==true) {
       cout << "Basis states for N-1 electrons (comma delimited):" << endl;
       vector<vector<int> > up_spin_sets_in_bit_form = get_up_spin_sets_in_bit_form();
       vector<vector<int> > down_spin_sets_in_bit_form = get_down_spin_sets_in_bit_form();
       int num_down = down_spin_sets_in_bit_form.size();
       int skip_m;
       if (up_spin_sets_in_bit_form.size() >= num_down) {
          skip_m = up_spin_sets_in_bit_form.size();
          for (int x = 0; x < num_down; x++) {
              for (int w = 0; w < up_spin_sets_in_bit_form.size(); w++) {
                  for (int up_index = 0; up_index < V; up_index++) {
                      cout << up_spin_sets_in_bit_form[w][up_index] << " ";
                  }
                  cout << "| ";
                  for (int down_index = 0; down_index < V; down_index++) {
                      cout << down_spin_sets_in_bit_form[x][down_index] << " ";
                  }
                  cout << ", ";
              }
          }
       }
       else {
            skip_m = num_down;
            for (int w = 0; w < up_spin_sets_in_bit_form.size(); w++) {
                for (int x = 0; x < num_down; x++) {
                    for (int up_index = 0; up_index < V; up_index++) {
                        cout << up_spin_sets_in_bit_form[w][up_index] << " ";
                    }
                    cout << "|";
                    for (int down_index = 0; down_index < V; down_index++) {
                        cout << down_spin_sets_in_bit_form[x][down_index] << " ";
                    }
                    cout << ", ";
                }
            }
       }
       cout << endl;
       cout << "Eigenvalues for N-1 electrons:" << endl;
       for (int i = 0; i < D2; i++) {
           cout << w_N_minus_one[i] << endl;
       }
       cout << "Eigenvectors (one per row) for N-1 electrons:" << endl;
       for (int j = 0; j < D2; j++) {
           for (int i = 0; i < D2; i++) {
               cout << z_N_minus_one[i*D2+j] << " ";
           }
           cout << endl;
       }
    }
    
    N_up = N_up_original + 1;
    N_down = N_down_original;
    combinations_up.clear();
    combinations_down.clear();
    int a[100];
    n = V;
    k = N_up;
    combinations_partial(a, 1, n, 1, k, true);
    n = V;
    k = N_down;
    int b[100];
    combinations_partial(b, 1, n, 1, k, false);
    fill_maps(N_up, N_down);
    convert_from_ket_to_index_of_part(N_up, N_down);
    int D3 = D_N_PLUS_ONE;
    matrix_for_T(N_up, N_down);
    matrix_for_U(N_up, N_down);
    add_matrices(D3);
    convert(D3,hamiltonian_N_plus_one);
    diagonalize(D3, ground_state_N_plus_one, hamiltonian_N_plus_one, z_N_plus_one, w_N_plus_one, ifail_N_plus_one);
    delete[] hamiltonian_N_plus_one;
    delete[] ground_state_N_plus_one;

    if (print==true) {
       cout << "Basis states for N+1 electrons (comma delimited):" << endl;
       vector<vector<int> > up_spin_sets_in_bit_form = get_up_spin_sets_in_bit_form();
       vector<vector<int> > down_spin_sets_in_bit_form = get_down_spin_sets_in_bit_form();
       int num_down = down_spin_sets_in_bit_form.size();
       int skip_l;
       if (up_spin_sets_in_bit_form.size() >= num_down) {
          skip_l = up_spin_sets_in_bit_form.size();
          for (int x = 0; x < num_down; x++) {
              for (int w = 0; w < up_spin_sets_in_bit_form.size(); w++) {
                  for (int up_index = 0; up_index < V; up_index++) {
                      cout << up_spin_sets_in_bit_form[w][up_index] << " ";
                  }
                  cout << "|";
                  for (int down_index = 0; down_index < V; down_index++) {
                      cout << down_spin_sets_in_bit_form[x][down_index] << " ";
                  }
                  cout << ", ";
              }
          }
       }
       else {
            skip_l = num_down;
            for (int w = 0; w < up_spin_sets_in_bit_form.size(); w++) {
                for (int x = 0; x < num_down; x++) {
                    for (int up_index = 0; up_index < V; up_index++) {
                        cout << up_spin_sets_in_bit_form[w][up_index] << " ";
                    }
                    cout << "|";
                    for (int down_index = 0; down_index < V; down_index++) {
                        cout << down_spin_sets_in_bit_form[x][down_index] << " ";
                    }
                    cout << ", ";
                }
            }
       }
       cout << endl;
       cout << "Eigenvalues for N+1 electrons:" << endl;
       for (int i = 0; i < D3; i++) {
           cout << w_N_plus_one[i] << endl;
       }
       cout << "Eigenvectors (one per row) for N+1 electrons:" << endl;
       for (int j = 0; j < D3; j++) {
           for (int i = 0; i < D3; i++) {
               cout << z_N_plus_one[i*D3+j] << " ";
           }
           cout << endl;
       }
    }
    

    cout << "Starting green's function calculations" << endl;
    for (int i = 0; i < number_of_time_points; i++) {
        times.push_back(i*time_step + start_time);
    }
    cout << "time vector formed" << endl;
    
    /* Shift Eigenvalues with the Chemical Potential */
    shift_eigenvalues(N_up_original, N_down_original); // CHEMICAL POTENTIAL
    
    
    output_values(N_up_original, N_down_original);
    
    cout << "frequency values outputted" << endl;
    
    output_ground_state_lesser_and_greater(N_up_original, N_down_original);
    
    cout << "lesser and greater values outputted" << endl;
    
    
    
    
    /* Construct a Polynomial */
    
    double my_K_value = 0;
    Symbolic my_denominator = construct_polynomial(N_up_original, N_down_original, my_K_value);
    

    /* Use coefficients */
    int degree = 47;
    const double coefficients [] = { 1.00000000000000 ,
        14.3154998244454 ,
        83.8818881178645 ,
        214.081798052430 ,
        42.8480586237278 ,
        -1055.17195959651 ,
        -1965.34710684436 ,
        1213.32105810016 ,
        7322.16436812661 ,
        3059.86443777955 ,
        -14290.9615729897 ,
        -13972.1456897929 ,
        15832.4280850507 ,
        24665.0119568897 ,
        -508.366240231704 ,
        -9710.47659144047 ,
        -46155.7985304933 ,
        -89422.8701642944 ,
        132938.255787390 ,
        375444.413324162 ,
        -237137.716490932 ,
        -965025.660918452 ,
        282825.921282616 ,
        1884534.45277379 ,
        -178623.430089721 ,
        -2946537.27712300 ,
        -77314.0392479293 ,
        3755611.13308910 ,
        337988.875700679 ,
        -3924334.71088915 ,
        -413796.411492566 ,
        3354167.14256408 ,
        262525.303530787 ,
        -2319281.56757682 ,
        -31833.2503086273 ,
        1267027.18145196 ,
        -105034.740241126 ,
        -522853.121727490 ,
        109158.242164974 ,
        149201.400820250 ,
        -55693.3205137050 ,
        -23601.5340229916 ,
        15472.3989005763 ,
        236.526468239246 ,
        -1800.05858529011 ,
        405.154854058112 ,
        -37.4607049405123 ,
        1.00000000000000 };
     
    
    // highest-order coefficient on the left
    // scientific notation example: -1.4406056132776368e+90
    
    
    complex<double> my_result [degree+1];
    cout << "error: " <<  newton_real( degree, coefficients, my_result ) << endl;
    for (int r = 1; r < degree + 1; r++) {
        complex<double> scaling_factor(0.274153683260457,0.0);
        my_result[r] = my_result[r];
        cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << my_result[r] * scaling_factor << endl;
    }
    
    
    
    delete[] z_N;
    delete[] z_N_minus_one;
    delete[] z_N_plus_one;
    delete[] matrix_m_c_n;
    delete[] matrix_n_c_dagger_m;
}






