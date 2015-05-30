#
# $Id: PDG.py,v 1.5 2009-01-26 03:05:43 ssnyder Exp $
# File: PDG.py
# Created: sss, Mar 2005
# Purpose: Define PDG ID codes.
#
"""
This module contains names for the various PDG particle ID codes.
The names are the same as in EventKernel/PdtPdg.h.

This module also contains a dictionary pdgid_names mapping ID codes
back to printable strings, and a function pdgid_to_name to do this
conversion.  Similarly, root_names and pdgid_to_root_name translate to
strings with root markup.
"""
from __future__ import absolute_import

from ROOT import TDatabasePDG
from pkg_resources import resource_filename
import os

db = TDatabasePDG()
db.ReadPDGTable(resource_filename('rootpy', 'etc/pdg_table.txt'))


def GetParticle(id):
    return db.GetParticle(id)

# Table to translate from PDG IDs to printable strings.
pdgid_names = {}

# Table to translate from PDG IDs to strings with root markup.
root_names = {}


def id_to_name(id):
    """
    Convert a PDG ID to a printable string.
    """
    name = pdgid_names.get(id)
    if not name:
        name = repr(id)
    return name


def id_to_root_name(id):
    """
    Convert a PDG ID to a string with root markup.
    """
    name = root_names.get(id)
    if not name:
        name = repr(id)
    return name

#
# Table of PDG IDs, associating the ID codes with up to several names.
# This is formatted as one big string to make it easier to maintain
# (don't need to quote everything individually).
# The format of each line is like this:
#
#    mname = id     pname   rname
#
# An attribute mname will be added to this module with a value of id.
# These names are intended to match those in PdgPdt.h.
# pname is a printable name for the entry, and rname is a name
# with root-style markup.  These names will be put into the pdgid_names
# and root_names dictionaries, respectively.  They can be left as `!'
# if no name is available.  pname and rname should not contain spaces.
# Blank lines or those starting with `#' will be ignored.
#
_pdgtable = \
"""
d = 1                                 D            d
anti_d = -1                           DBAR         #bar{d}
u = 2                                 U            u
anti_u = -2                           UBAR         #bar{u}
s = 3                                 S            s
anti_s = -3                           SBAR         #bar{s}
c = 4                                 C            c
anti_c = -4                           CBAR         #bar{c}
b = 5                                 B            b
anti_b = -5                           BBAR         #bar{b}
t = 6                                 T            t
anti_t = -6                           TBAR         #bar{t}
l = 7                                 LPRIME       !
anti_l = -7                           LPRIMEBAR    !
h = 8                                 !            !
anti_h = -8                           !            !
g = 21                                GLUE         g
e_minus = 11                          E-           e^{-}
e_plus = -11                          E+           e^{+}
nu_e = 12                             NUE          #nu_{e}
anti_nu_e = -12                       ANUE         #bar{#nu}_{e}
mu_minus = 13                         MU-          #mu^{-}
mu_plus = -13                         MU+          #mu^{+}
nu_mu = 14                            NUM          #nu_{#mu}
anti_nu_mu = -14                      ANUM         #bar{#nu}_{#mu}
tau_minus = 15                        TAU-         #tau^{-}
tau_plus = -15                        TAU+         #tau^{+}
nu_tau = 16                           NUT          #nu_{#tau}
anti_nu_tau = -16                     ANUT         #bar{nu}_{#tau}
L_minus = 17                          !            !
L_plus = -17                          !            !
nu_L = 18                             !            !
anti_nu_L = -18                       !            !
gamma = 22                            PHOT         #gamma
Z0 = 23                               Z0           Z
W_plus = 24                           W+           W^{+}
W_minus = -24                         W-           W^{-}
Higgs0 = 25                           H0           h^{0}
reggeon = 28                          !            !
pomeron = 29                          !            !
Z_prime0 = 32                         !            !
Z_prime_prime0 = 33                   !            !
W_prime_plus = 34                     !            !
W_prime_minus = -34                   !            !
Higgs_prime0 = 35                     !            !
A0 = 36                               !            !
Higgs_plus = 37                       !            !
Higgs_minus = -37                     !            !
R0 = 40                               !            !
anti_R0 = -40                         !            !
specflav = 81                         !            !
rndmflav = 82                         !            !
anti_rndmflav = -82                   !            !
phasespa = 83                         !            !
c_minushadron = 84                    !            !
anti_c_minushadron = -84              !            !
b_minushadron = 85                    !            !
anti_b_minushadron = -85              !            !
t_minushadron = 86                    !            !
anti_t_minushadron = -86              !            !
Wvirt_plus = 89                       !            !
Wvirt_minus = -89                     !            !
diquark = 90                          !            !
anti_diquark = -90                    !            !
cluster = 91                          CLUSTER      cluster
string = 92                           !            !
indep = 93                            !            !
CMshower = 94                         !            !
SPHEaxis = 95                         !            !
THRUaxis = 96                         !            !
CLUSjet = 97                          !            !
CELLjet = 98                          !            !
table = 99                            !            !
pi0 = 111                             PI0          #pi^{0}
pi_plus = 211                         PI+          #pi^{+}
pi_minus = -211                       PI-          #pi^{-}
pi_diffr_plus = 210                   !            !
pi_diffr_minus = -210                 !            !
pi_2S0 = 20111                        !            !
pi_2S_plus = 20211                    !            !
pi_2S_minus = -20211                  !            !
eta = 221                             ETA          #eta
eta_2S = 20221                        !            !
eta_prime = 331                       !            !
rho0 = 113                            !            #rho^{0}
rho_plus = 213                        RHO+         #rho^{+}
rho_minus = -213                      RHO-         #rho^{-}
rho_2S0 = 30113                       !            !
rho_2S_plus = 30213                   !            !
rho_2S_minus = -30213                 !            !
rho_3S0 = 40113                       !            !
rho_3S_plus = 40213                   !            !
rho_3S_minus = -40213                 !            !
omega = 223                           !            !
omega_2S = 30223                      !            !
phi = 333                             PHI          #phi
a_00 = 10111                          !            !
a_0_plus = 10211                      !            !
a_0_minus = -10211                    !            !
f_0 = 10221                           !            !
f_prime_0 = 10331                     !            !
b_10 = 10113                          !            !
b_1_plus = 10213                      !            !
b_1_minus = -10213                    !            !
h_1 = 10223                           h_1          h_{1}
h_prime_1 = 10333                     !            !
a_10 = 20113                          !            !
a_1_plus = 20213                      !            !
a_1_minus = -20213                    !            !
f_1 = 20223                           !            !
f_prime_1 = 20333                     !            !
a_20 = 115                            !            !
a_2_plus = 215                        a_2+         a_{2}^{+}
a_2_minus = -215                      a_2-         a_{2}^{-}
f_2 = 225                             !            !
f_prime_2 = 335                       !            !
K0 = 311                              K0           K^{0}
anti_K0 = -311                        K0BAR        #bar{K}^0
K_S0 = 310                            K_S0         K_{S}^{0}
K_L0 = 130                            K_L0         K_{L}^{0}
K_plus = 321                          K+           K^{+}
K_minus = -321                        K-           K^{-}
K_star0 = 313                         K*           K^{*}
anti_K_star0 = -313                   K*BAR        #bar{K}^{*}
K_star_plus = 323                     !            !
K_star_minus = -323                   !            !
K_0_star0 = 10311                     !            !
anti_K_0_star0 = -10311               !            !
K_0_star_plus = 10321                 !            !
K_0_star_minus = -10321               !            !
K_10 = 10313                          !            !
anti_K_10 = -10313                    !            !
K_1_plus = 10323                      !            !
K_1_minus = -10323                    !            !
K_2_star0 = 315                       !            !
anti_K_2_star0 = -315                 !            !
K_2_star_plus = 325                   K_2*+        K_{2}^{*+}
K_2_star_minus = -325                 K_2*-        K_{2}^{*-}
K_prime_10 = 20313                    !            !
anti_K_prime_10 = -20313              !            !
K_prime_1_plus = 20323                !            !
K_prime_1_minus = -20323              !            !
D_plus = 411                          D+           D^{+}
D_minus = -411                        D-           D^{-}
D0 = 421                              D0           D^{0}
anti_D0 = -421                        D0BAR        #bar{D}^{0}
D_star_plus = 413                     !            !
D_star_minus = -413                   !            !
D_star0 = 423                         !            !
anti_D_star0 = -423                   !            !
D_0_star_plus = 10411                 !            !
D_0_star_minus = -10411               !            !
D_0_star0 = 10421                     !            !
anti_D_0_star0 = -10421               !            !
D_1_plus = 10413                      !            !
D_1_minus = -10413                    !            !
D_10 = 10423                          !            !
anti_D_10 = -10423                    !            !
D_2_star_plus = 415                   !            !
D_2_star_minus = -415                 !            !
D_2_star0 = 425                       !            !
anti_D_2_star0 = -425                 !            !
D_prime_1_plus = 20413                !            !
D_prime_1_minus = -20413              !            !
D_prime_10 = 20423                    !            !
anti_D_prime_10 = -20423              !            !
D_s_plus = 431                        D_S+         D_{s}^{+}
D_s_minus = -431                      D_S-         D_{s}^{-}
D_s_star_plus = 433                   !            !
D_s_star_minus = -433                 !            !
D_s0_star_plus = 10431                !            !
D_s0_star_minus = -10431              !            !
D_s1_plus = 10433                     !            !
D_s1_minus = -10433                   !            !
D_s2_star_plus = 435                  !            !
D_s2_star_minus = -435                !            !
D_prime_s1_plus = 20433               !            !
D_prime_s1_minus = -20433             !            !
B0 = 511                              B0           B^{0}
anti_B0 = -511                        B0BAR        #bar{B}^{0}
B_plus = 521                          B+           B^{+}
B_minus = -521                        B-           B^{-}
B_star0 = 513                         !            !
anti_B_star0 = -513                   !            !
B_star_plus = 523                     !            !
B_star_minus = -523                   !            !
B_0_star0 = 10511                     !            !
anti_B_0_star0 = -10511               !            !
B_0_star_plus = 10521                 !            !
B_0_star_minus = -10521               !            !
B_10 = 10513                          !            !
anti_B_10 = -10513                    !            !
B_1_plus = 10523                      !            !
B_1_minus = -10523                    !            !
B_2_star0 = 515                       !            !
anti_B_2_star0 = -515                 !            !
B_2_star_plus = 525                   !            !
B_2_star_minus = -525                 !            !
B_prime_10 = 20513                    !            !
anti_B_prime_10 = -20513              !            !
B_prime_1_plus = 20523                !            !
B_prime_1_minus = -20523              !            !
B_s0 = 531                            B_S0         B_{s}^{0}
anti_B_s0 = -531                      B_S0BAR      #bar{B}_{s}^{0}
B_s_star0 = 533                       !            !
anti_B_s_star0 = -533                 !            !
B_s0_star0 = 10531                    !            !
anti_B_s0_star0 = -10531              !            !
B_s10 = 10533                         !            !
anti_B_s10 = -10533                   !            !
B_s2_star0 = 535                      !            !
anti_B_s2_star0 = -535                !            !
B_prime_s10 = 20533                   !            !
anti_B_prime_s10 = -20533             !            !
B_c_plus = 541                        BC+          B_{c}^{+}
B_c_minus = -541                      BC-          B_{c}^{-}
B_c_star_plus = 543                   BC*+         B_{c}^{*+}
B_c_star_minus = -543                 BC*-         B_{c}^{*-}
B_c0_star_plus = 10541                !            !
B_c0_star_minus = -10541              !            !
B_c1_plus = 10543                     !            !
B_c1_minus = -10543                   !            !
B_c2_star_plus = 545                  !            !
B_c2_star_minus = -545                !            !
B_prime_c1_plus = 20543               !            !
B_prime_c1_minus = -20543             !            !
eta_c = 441                           !            !
eta_c_2S = 20441                      !            !
J_psi = 443                           JPSI         J/#psi
psi_2S = 20443                        !            !
chi_c0 = 10441                        !            !
chi_c1 = 10443                        !            !
chi_c2 = 445                          !            !
eta_b_2S = 20551                      !            !
eta_b_3S = 40551                      !            !
Upsilon = 553                         !            !
Upsilon_2S = 20553                    !            !
Upsilon_3S = 60553                    !            !
Upsilon_4S = 70553                    !            !
Upsilon_5S = 80553                    !            !
h_b = 10553                           !            !
h_b_2P = 40553                        !            !
h_b_3P = 100553                       !            !
chi_b0 = 551                          !            !
chi_b1 = 20553                        !            !
chi_b2 = 555                          !            !
chi_b0_2P = 30551                     !            !
chi_b1_2P = 50553                     !            !
chi_b2_2P = 10555                     !            !
chi_b0_3P = 50551                     !            !
chi_b1_3P = 110553                    !            !
chi_b2_3P = 20555                     !            !
eta_b2_1D = 40555                     !            !
eta_b2_2D = 60555                     !            !
Upsilon_1_1D = 120553                 !            !
Upsilon_2_1D = 30555                  !            !
Upsilon_3_1D = 557                    !            !
Upsilon_1_2D = 130553                 !            !
Upsilon_2_2D = 50555                  !            !
Upsilon_3_2D = 10557                  !            !
Delta_minus = 1114                    DELTA-       #Delta^{-}
anti_Delta_plus = -1114               DELTA+       #Delta^{+}
n_diffr = 2110                        !            !
anti_n_diffr = -2110                  !            !
n0 = 2112                             N            n
anti_n0 = -2112                       NBAR         #bar{n}
Delta0 = 2114                         !            !
anti_Delta0 = -2114                   !            !
p_diffr_plus = 2210                   !            !
anti_p_diffr_minus = -2210            !            !
p_plus = 2212                         P+           p^{+}
anti_p_minus = -2212                  P-           p^{-}
Delta_plus = 2214                     !            !
anti_Delta_minus = -2214              !            !
Delta_plus_plus = 2224                !            !
anti_Delta_minus_minus = -2224        !            !
Sigma_minus = 3112                    SIGMA-       #Sigma^{-}
anti_Sigma_plus = -3112               SIGMABAR+    #bar{#Sigma}^{+}
Sigma_star_minus = 3114               !            !
anti_Sigma_star_plus = -3114          !            !
Lambda0 = 3122                        LAMBDA_D0    #Lambda^{0}
anti_Lambda0 = -3122                  LAMBDABAR_D0 #bar{#Lambda}^{0}
Sigma0 = 3212                         !            !
anti_Sigma0 = -3212                   !            !
Sigma_star0 = 3214                    !            !
anti_Sigma_star0 = -3214              !            !
Sigma_plus = 3222                     SIGMA+       #Sigma^{+}
anti_Sigma_minus = -3222              SIGMABAR-    #bar{#Sigma}^{-}
Sigma_star_plus = 3224                !            !
anti_Sigma_star_minus = -3224         !            !
Xi_minus = 3312                       XI-          #Xi^{-}
anti_Xi_plus = -3312                  XI+          #Xi^{+}
Xi_star_minus = 3314                  !            !
anti_Xi_star_plus = -3314             !            !
Xi0 = 3322                            XI0          #Xi^{0}
anti_Xi0 = -3322                      XIBAR0       #bar{Xi}^{0}
Xi_star0 = 3324                       !            !
anti_Xi_star0 = -3324                 !            !
Omega_minus = 3334                    !            !
anti_Omega_plus = -3334               !            !
Sigma_c0 = 4112                       !            !
anti_Sigma_c0 = -4112                 !            !
Sigma_c_star0 = 4114                  SIGMA_C0*    #Sigma_{c}^{*0}
anti_Sigma_c_star0 = -4114            SIGMABAR_C0* #bar{#Sigma}_{c}^{*0}
Lambda_c_plus = 4122                  LAMBDA_C+    #Lambda_{c}^{+}
anti_Lambda_c_minus = -4122           LAMBDA_C-    #Lambda_{c}^{-}
Xi_c0 = 4132                          XI_C0        #Xi_{c}^{0}
anti_Xi_c0 = -4132                    XIBAR_C0     #bar{#Xi}_{c}^{0}
Sigma_c_plus = 4212                   SIGMA_C+     #Sigma_{c}^{+}
anti_Sigma_c_minus = -4212            SIGMA_C-     #Sigma_{c}^{-}
Sigma_c_star_plus = 4214              SIGMA_C+*    #Sigma_{c}^{*+}
anti_Sigma_c_star_minus = -4214       SIGMA_C-*    #Sigma_{c}^{*-}
Sigma_c_plus_plus = 4222              SIGMA_C++    #Sigma_{c}^{++}
anti_Sigma_c_minus_minus = -4222      SIGMA_C--    #Sigma_{c}^{--}
Sigma_c_star_plus_plus = 4224         SIGMA_C++*   #Sigma_{c}^{*++}
anti_Sigma_c_star_minus_minus = -4224 SIGMA_C--*   #Sigma_{c}^{*--}
Xi_c_plus = 4322                      XI_C+        #Xi_{c}^{+}
anti_Xi_c_minus = -4322               XI_C-        #Xi_{c}^{-}
Xi_prime_c0 = 4312                    XI'_C0       #Xi\'_{c}^{0}
Xi_primeanti__c0 = -4312              XIBAR'_C0    #bar{#Xi}\'_{c}^{0}
Xi_c_star0 = 4314                     XI_C0*       #Xi_{c}^{*0}
anti_Xi_c_star0 = -4314               XIBAR_C0*    #bar{#Xi}_{c}^{*0}
Xi_prime_c_plus = 4232                XI'_C+       #Xi\'_{c}^{+}
Xi_primeanti__c_minus = -4232         XIBAR'_C-    #Xi\'_{c}^{-}
Xi_c_star_plus = 4324                 XI_C+*       #Xi_{c}^{*+}
anti_Xi_c_star_minus = -4324          XI_C-*       #Xi_{c}^{*-}
Omega_c0 = 4332                       OMEGA_C0     #Omega_{c}^{0}
anti_Omega_c0 = -4332                 OMEGABAR_C0  #bar{#Omega}_{c}^{0}
Omega_c_star0 = 4334                  OMEGA_C0*    #Omega_{c}^{*0}
anti_Omega_c_star0 = -4334            OMEGA_C0*    #bar{#Omega}_{c}^{*0}
Sigma_b_minus = 5112                  SIGMA_B-     #Sigma_{b}^{-}'
anti_Sigma_b_plus = -5112             SIGMA_B+     #Sigma_{b}^{+}'
Sigma_b_star_minus = 5114             !            !
anti_Sigma_b_star_plus = -5114        !            !
Lambda_b0 = 5122                      LAMBDA_B0    #Lambda_{b}^{0}
anti_Lambda_b0 = -5122                LAMBDA_B0BAR #bar{#Lambda}_{b}^0
Xi_b_minus = 5132                     !            !
anti_Xi_b_plus = -5132                !            !
Sigma_b0 = 5212                       SIGMA_B0     #Sigma_{b}^{0}
anti_Sigma_b0 = -5212                 SIGMABAR_B0  #bar{#Sigma}_{b}^{0}
Sigma_b_star0 = 5214                  !            !
anti_Sigma_b_star0 = -5214            !            !
Sigma_b_plus = 5222                   !            !
anti_Sigma_b_minus = -5222            !            !
Sigma_star_ = 5224                    !            !
anti_Sigma_b_star_minus = -5224       !            !
Xi_b0 = 5232                          XI_B0        #Xi_b^{0}
anti_Xi_b0 = -5232                    XIBAR_B0     #bar{#Xi}_b^{0}
Xi_prime_b_minus = 5312               !            !
anti_Xi_prime_b_plus = -5312          !            !
Xi_b_star_minus = 5314                !            !
anti_Xi_b_star_plus = -5314           !            !
Xi_prime_b0 = 5322                    !            !
anti_Xi_prime_b0 = -5322              !            !
Xi_b_star0 = 5324                     !            !
anti_Xi_b_star0 = -5324               !            !
Omega_b_minus = 5332                  !            !
anti_Omega_b_plus = -5332             !            !
Omega_b_star_minus = 5334             !            !
anti_Omega_b_star_plus = -5334        !            !
dd_0 = 1101                           !            !
anti_dd_0 = -1101                     !            !
ud_0 = 2101                           UD0          !
anti_ud_0 = -2101                     UD0BAR       !
uu_0 = 2201                           !            !
anti_uu_0 = -2201                     !            !
sd_0 = 3101                           !            !
anti_sd_0 = -3101                     !            !
su_0 = 3201                           !            !
anti_su_0 = -3201                     !            !
ss_0 = 3301                           !            !
anti_ss_0 = -3301                     !            !
cd_0 = 4101                           !            !
anti_cd_0 = -4101                     !            !
cu_0 = 4201                           !            !
anti_cu_0 = -4201                     !            !
cs_0 = 4301                           !            !
anti_cs_0 = -4301                     !            !
cc_0 = 4401                           !            !
anti_cc_0 = -4401                     !            !
bd_0 = 5101                           !            !
anti_bd_0 = -5101                     !            !
bu_0 = 5201                           !            !
anti_bu_0 = -5201                     !            !
bs_0 = 5301                           !            !
anti_bs_0 = -5301                     !            !
bc_0 = 5401                           !            !
anti_bc_0 = -5401                     !            !
bb_0 = 5501                           !            !
anti_bb_0 = -5501                     !            !
dd_1 = 1103                           !            !
anti_dd_1 = -1103                     !            !
ud_1 = 2103                           !            !
anti_ud_1 = -2103                     !            !
uu_1 = 2203                           !            !
anti_uu_1 = -2203                     !            !
sd_1 = 3103                           !            !
anti_sd_1 = -3103                     !            !
su_1 = 3203                           !            !
anti_su_1 = -3203                     !            !
ss_1 = 3303                           !            !
anti_ss_1 = -3303                     !            !
cd_1 = 4103                           !            !
anti_cd_1 = -4103                     !            !
cu_1 = 4203                           !            !
anti_cu_1 = -4203                     !            !
cs_1 = 4303                           !            !
anti_cs_1 = -4303                     !            !
cc_1 = 4403                           !            !
anti_cc_1 = -4403                     !            !
bd_1 = 5103                           !            !
anti_bd_1 = -5103                     !            !
bu_1 = 5203                           !            !
anti_bu_1 = -5203                     !            !
bs_1 = 5303                           !            !
anti_bs_1 = -5303                     !            !
bc_1 = 5403                           !            !
anti_bc_1 = -5403                     !            !
bb_1 = 5503                           !            !
anti_bb_1 = -5503                     !            !

# SUSY Particles names modified from /Control/AthenaCommon/PDGTABLE.MeV
# naming convention change
#      '~' to 's_'
#      '(' to '_'
#      ')' to nothing
#      '+' to 'plus'
#      '' to '_'
#      for the negatively charged particles so I add "minus" to the name and a corresponding "plus" entry with -pdg code
#      for the neutrals I add a corresponding "anti" entry with -pdg code
#      for the particles with positive charge entries I add a corresponding "minus" entry with -pdg code
# ************ (the above is not consistent with the convention that minus=particle plus=anti-particle
#
#      Next remove Majorana particles and rename L-R stau to mass eigenstates.
#
#      This is all ugly but sort of consistent with previous naming convention

s_e_minus_L    =1000011               !            !
s_e_plus_L     =-1000011              !            !

s_nu_e_L       =1000012               !            !
s_anti_nu_e_L  =-1000012              !            !

s_mu_minus_L   =1000013               !            !
s_mu_plus_L    =-1000013              !            !

s_nu_mu_L      =1000014               !            !
s_anti_nu_mu_L =-1000014              !            !

#    s_tau_minus_L  =1000015
#    s_tau_plus_L   =-1000015

# L-R mixing significant use _1 and _2 for names instead
s_tau_minus_1  =1000015               !            !
s_tau_plus_1   =-1000015              !            !

s_nu_tau_L     =1000016               !            !
s_anti_nu_tau_L=-1000016              !            !

s_e_minus_R    =2000011               !            !
s_e_plus_R     =-2000011              !            !

s_mu_minus_R   =2000013               !            !
s_mu_plus_R    =-2000013              !            !

s_tau_minus_2  =2000015               !            !
s_tau_plus_2   =-2000015              !            !

s_g            =1000021               !            !
#    s_anti_g       =-1000021 # Majorana

s_chi_0_1      =1000022               !            !
#    s_anti_chi_0_1 =-1000022 # Majorana

s_chi_0_2      =1000023               !            !
#    s_anti_chi_0_2 =-1000023 # Majorana

s_chi_plus_1   =1000024               !            !
# Majorana
s_chi_minus_1  =-1000024              !            !

s_chi_0_3      =1000025               !            !
#    s_anti_chi_0_3 =-1000025 # Majorana

s_chi_0_4      =1000035               !            !
#    s_anti_chi_0_4 =-1000035 # Majorana

s_chi_plus_2   =1000037               !            !
s_chi_minus_2  =-1000037              !            !

s_G            =1000039               !            !
#    s_anti_G       =-1000039 # Majorana

# note mismatch with PDGTable and pre-existing PdtPdg.h
#M     999                          0.E+00         +0.0E+00 -0.0E+00 Geantino        0
#W     999                          0.E+00         +0.0E+00 -0.0E+00 Geantino        0

# doubly charged Higgs
Higgs_plus_plus_L = 9900041           !            !
Higgs_minus_minus_L = -9900041        !            !
Higgs_plus_plus_R = 9900042           !            !
Higgs_minus_minus_R = -9900042        !            !


# Null particles
deuteron = 0                          !            !
tritium = 0                           !            !
alpha = 0                             !            !
geantino = 0                          !            !
He3 = 0                               !            !
Cerenkov = 0                          !            !
null = 0                              !            !


# Some extra particles that weren't in PdgPdt.h
Xi_cc_plus = 4412                     XI_CC+       #Xi_{cc}^{+}
anti_Xi_cc_minus = -4412              XI_CC-       #Xi_{cc}^{-}
Xi_cc_plus_plus = 4422                XI_CC++      #Xi_{cc}^{++}
anti_Xi_cc_minus_minus = -4422        XI_CC--      #Xi_{cc}^{--}
Xi_cc_star_plus = 4414                XI_CC+*      #Xi_{cc}^{*+}
anti_Xi_cc_star_minus = -4414         XI_CC-*      #Xi_{cc}^{*-}
Xi_cc_star_plus_plus = 4424           XI_CC++*     #Xi_{cc}^{*++}
anti_Xi_cc_star_minus_minus = -4424   XI_CC--*     #Xi_{cc}^{*--}
Omega_cc_plus = 4432                  OMEGA_CC+    #Omega_{cc}^{+}
anti_Omega_cc_minus = -4432           OMEGA_CC-    #Omega_{cc}^{-}
Omega_cc_star_plus = 4434             OMEGA_CC+*   #Omega_{cc}^{*+}
anti_Omega_cc_star_minus = -4434      OMEGA_CC-*   #Omega_{cc}^{*-}
Omega_ccc_plus_plus = 4444            OMEGA_CCC++  #Omega_{ccc}^{++}
anti_Omega_ccc_minus_minus = -4444    OMEGA_CCC--  #Omega_{ccc}^{--}


# A couple extra synonyms that weren't in PdgPdt.h.
e = e_minus                           !            !
mu = mu_minus                         !            !
tau = tau_minus                       !            !
W = W_plus                            !            !
"""


# Parse _pdgtable and fill in dictionaries.
def _fill_dicts():
    import string
    pdgid_names.clear()
    root_names.clear()
    for line in _pdgtable.split ('\n'):
        line = line.strip()
        if len(line) == 0 or line[0] == '#': continue
        ll = line.split('=', 1)
        if len(ll) < 2:
            print('bad line: {0}'.format(line))
            continue
        mname = string.strip(ll[0])
        ll = ll[1].split()
        if len(ll) < 1:
            print('bad line: {0}'.format(line))
            continue
        id = ll[0]
        pname = None
        if len(ll) >= 2 and ll[1] != '!':
            pname = ll[1]
        rname = None
        if len(ll) >= 3 and ll[2] != '!':
            rname = ll[2]
        try:
            id = int(id)
        except ValueError:
            id = globals().get(id)
            if id == None:
                print('bad line: {0}'.format(line))
                continue

        if pname == None:
            pname = mname
        if rname == None:
            rname = pname

        globals()[mname] = id
        if not pdgid_names.has_key(id):
            pdgid_names[id] = pname
        if not root_names.has_key(id):
            root_names[id] = rname
    return

# Fill the dictionaries.
_fill_dicts()

# Kill these now to save memory.
del _pdgtable
del _fill_dicts
