#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_35(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_5667407676628393486);
void live_err_fun(double *nom_x, double *delta_x, double *out_8668485348903095543);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_5202107570088414963);
void live_H_mod_fun(double *state, double *out_6130977875155903016);
void live_f_fun(double *state, double dt, double *out_2092095269018721650);
void live_F_fun(double *state, double dt, double *out_6131485590610354164);
void live_h_4(double *state, double *unused, double *out_9169265721936850834);
void live_H_4(double *state, double *unused, double *out_8583382569498834716);
void live_h_9(double *state, double *unused, double *out_1190771968094934266);
void live_H_9(double *state, double *unused, double *out_1296163634234387246);
void live_h_10(double *state, double *unused, double *out_518897998535379324);
void live_H_10(double *state, double *unused, double *out_3441208344363766057);
void live_h_12(double *state, double *unused, double *out_5953485100376480286);
void live_H_12(double *state, double *unused, double *out_3563926161466872921);
void live_h_35(double *state, double *unused, double *out_4415523909377844667);
void live_H_35(double *state, double *unused, double *out_5216720512126227340);
void live_h_32(double *state, double *unused, double *out_1202921583661580157);
void live_H_32(double *state, double *unused, double *out_584583860081385218);
void live_h_13(double *state, double *unused, double *out_5949023650374730472);
void live_H_13(double *state, double *unused, double *out_8461368207358977669);
void live_h_14(double *state, double *unused, double *out_1190771968094934266);
void live_H_14(double *state, double *unused, double *out_1296163634234387246);
void live_h_33(double *state, double *unused, double *out_4507703079326906254);
void live_H_33(double *state, double *unused, double *out_2066163507487369736);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}