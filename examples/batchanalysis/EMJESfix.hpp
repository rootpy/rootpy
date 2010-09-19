///******************************************
///******************************************
///
/// A temporary routine for fixing null EMJES moment.
/// 
/// This is *temporary* and should *NOT* be applied on analysis files
/// built with athena > 15.6.8.9
///
/// Usage in C++
///   1) include this header in your analysis code
///   2) instantiate a EMJESFixer object once :
///  EMJESFixer jetEMJESfixer;
///
///   3) use it if the EMJES moment is null (this example for AntiKt4H1TopoJets)   :
///
///  double emjes = ...; // retrieve EMJES from jet or ntuple
///  if( emjes == 0 ){
///    emjes = jetEMJESfixer.fixAntiKt4H1Topo(jet_pt,jet_eta);
///  }
///  // use emjes as you wish
///
/// Usage in python
///  1) Load the macro and instantiate an object :
///
/// ROOT.gSystem.CompileMacro( 'EMJESfix.hpp')
/// jetEMJESfixer = ROOT.EMJESFixer()
///
/// 2) use it if the EMJES moment is null (this example for AntiKt4H1TopoJets)  
///
/// # retrieve EMJES from jet or ntuple in emjes
/// if emjes == 0:
///     emjes = jetEMJESfixer.fixAntiKt4H1Topo(jet_pt,jet_eta)
/// # use emjes as you wish
///
///******************************************

#include <cmath>

class EMJESFixer {

    public:
        EMJESFixer(){
            fillConstants();
        }



        double fix(double jet_pt, double jet_eta, double  calibConstants[][4])
        {
            int index = 0;
            double R=-1;

            for(unsigned int i=0; i< 44; i++)
                if(fabs(jet_eta) > (i+1)*0.1)
                    index = i+1;

            // Now obtain the calibration constants
            double a0 = 1;
            double a1 = calibConstants[index][0];
            double a2 = calibConstants[index][1];
            double a3 = calibConstants[index][2];
            double a4 = calibConstants[index][3];

            double m_ptcut = 10;
            double GeV = 1000;

            // Calculate R term by term
            R = a0;

            // Apply pT cut
            if(jet_pt < m_ptcut*GeV) jet_pt = m_ptcut*GeV;

            // Use Jet pT in GeV
            double log_jet_pt = log(jet_pt/GeV);

            // Protect vs. division by zero
            if(log_jet_pt != 0)
            {
                R += a1/log_jet_pt;
                R += a2/pow(log_jet_pt,2);
                R += a3/pow(log_jet_pt,3);
                R += a4/pow(log_jet_pt,4);
            }
            return 1./R;
        }

        double fixAntiKt6H1Tower(double jet_pt, double jet_eta){
            return fix(jet_pt , jet_eta , m_AntiKt6H1Tower);
        }

        double fixAntiKt6H1Topo(double jet_pt, double jet_eta){
            return fix(jet_pt , jet_eta, m_AntiKt6H1Topo );
        }


        double fixAntiKt4H1Tower(double jet_pt, double jet_eta){
            return fix(jet_pt , jet_eta , m_AntiKt4H1Tower);
        }

        double fixAntiKt4H1Topo(double jet_pt, double jet_eta){
            return fix(jet_pt , jet_eta, m_AntiKt4H1Topo );
        }




        double m_AntiKt6H1Tower[45][4];
        double m_AntiKt6H1Topo[45][4];
        double m_AntiKt4H1Tower[45][4];
        double m_AntiKt4H1Topo[45][4];




        void fillConstants(){

            // fill constants
            // Nasty: Put all the constants in
            m_AntiKt6H1Topo[0][0] = -0.676568; m_AntiKt6H1Topo[0][1] = -6.73995; m_AntiKt6H1Topo[0][2] = 21.8839; m_AntiKt6H1Topo[0][3] = -19.9172; 
            m_AntiKt6H1Topo[1][0] = -0.645147; m_AntiKt6H1Topo[1][1] = -6.25005; m_AntiKt6H1Topo[1][2] = 18.6959; m_AntiKt6H1Topo[1][3] = -15.3563; 
            m_AntiKt6H1Topo[2][0] = -0.48034; m_AntiKt6H1Topo[2][1] = -7.83875; m_AntiKt6H1Topo[2][2] = 24.2973; m_AntiKt6H1Topo[2][3] = -22.0308; 
            m_AntiKt6H1Topo[3][0] = -0.518052; m_AntiKt6H1Topo[3][1] = -7.44206; m_AntiKt6H1Topo[3][2] = 22.7878; m_AntiKt6H1Topo[3][3] = -20.0835; 
            m_AntiKt6H1Topo[4][0] = -0.47104; m_AntiKt6H1Topo[4][1] = -7.51981; m_AntiKt6H1Topo[4][2] = 21.7959; m_AntiKt6H1Topo[4][3] = -17.9213; 
            m_AntiKt6H1Topo[5][0] = -0.469625; m_AntiKt6H1Topo[5][1] = -7.85256; m_AntiKt6H1Topo[5][2] = 23.6274; m_AntiKt6H1Topo[5][3] = -20.6898; 
            m_AntiKt6H1Topo[6][0] = -0.297181; m_AntiKt6H1Topo[6][1] = -9.5623; m_AntiKt6H1Topo[6][2] = 28.5822; m_AntiKt6H1Topo[6][3] = -25.3111; 
            m_AntiKt6H1Topo[7][0] = -0.375276; m_AntiKt6H1Topo[7][1] = -9.98404; m_AntiKt6H1Topo[7][2] = 32.0187; m_AntiKt6H1Topo[7][3] = -30.423; 
            m_AntiKt6H1Topo[8][0] = -0.821604; m_AntiKt6H1Topo[8][1] = -5.94113; m_AntiKt6H1Topo[8][2] = 18.547; m_AntiKt6H1Topo[8][3] = -15.1776; 
            m_AntiKt6H1Topo[9][0] = -1.10474; m_AntiKt6H1Topo[9][1] = -3.72317; m_AntiKt6H1Topo[9][2] = 13.0692; m_AntiKt6H1Topo[9][3] = -11.2257; 
            m_AntiKt6H1Topo[10][0] = -0.057028; m_AntiKt6H1Topo[10][1] = -12.5764; m_AntiKt6H1Topo[10][2] = 38.5174; m_AntiKt6H1Topo[10][3] = -35.7054; 
            m_AntiKt6H1Topo[11][0] = 0.456398; m_AntiKt6H1Topo[11][1] = -16.619; m_AntiKt6H1Topo[11][2] = 49.7; m_AntiKt6H1Topo[11][3] = -46.4188; 
            m_AntiKt6H1Topo[12][0] = 0.191065; m_AntiKt6H1Topo[12][1] = -14.4411; m_AntiKt6H1Topo[12][2] = 43.3863; m_AntiKt6H1Topo[12][3] = -40.1411; 
            m_AntiKt6H1Topo[13][0] = -0.670523; m_AntiKt6H1Topo[13][1] = -7.80262; m_AntiKt6H1Topo[13][2] = 24.8267; m_AntiKt6H1Topo[13][3] = -21.8666; 
            m_AntiKt6H1Topo[14][0] = -1.59944; m_AntiKt6H1Topo[14][1] = -0.754052; m_AntiKt6H1Topo[14][2] = 6.83622; m_AntiKt6H1Topo[14][3] = -6.6323; 
            m_AntiKt6H1Topo[15][0] = -0.781707; m_AntiKt6H1Topo[15][1] = -4.59825; m_AntiKt6H1Topo[15][2] = 11.5363; m_AntiKt6H1Topo[15][3] = -6.32582; 
            m_AntiKt6H1Topo[16][0] = -0.580922; m_AntiKt6H1Topo[16][1] = -5.37769; m_AntiKt6H1Topo[16][2] = 13.6154; m_AntiKt6H1Topo[16][3] = -9.0677; 
            m_AntiKt6H1Topo[17][0] = -0.274334; m_AntiKt6H1Topo[17][1] = -6.85484; m_AntiKt6H1Topo[17][2] = 16.0482; m_AntiKt6H1Topo[17][3] = -9.96231; 
            m_AntiKt6H1Topo[18][0] = -0.210738; m_AntiKt6H1Topo[18][1] = -7.2436; m_AntiKt6H1Topo[18][2] = 18.2536; m_AntiKt6H1Topo[18][3] = -13.5124; 
            m_AntiKt6H1Topo[19][0] = -0.142391; m_AntiKt6H1Topo[19][1] = -7.59879; m_AntiKt6H1Topo[19][2] = 19.066; m_AntiKt6H1Topo[19][3] = -13.9389; 
            m_AntiKt6H1Topo[20][0] = -0.371412; m_AntiKt6H1Topo[20][1] = -5.36762; m_AntiKt6H1Topo[20][2] = 12.3751; m_AntiKt6H1Topo[20][3] = -7.20744; 
            m_AntiKt6H1Topo[21][0] = -0.403531; m_AntiKt6H1Topo[21][1] = -4.99653; m_AntiKt6H1Topo[21][2] = 11.7236; m_AntiKt6H1Topo[21][3] = -6.91864; 
            m_AntiKt6H1Topo[22][0] = -0.23515; m_AntiKt6H1Topo[22][1] = -6.75943; m_AntiKt6H1Topo[22][2] = 18.5876; m_AntiKt6H1Topo[22][3] = -15.2072; 
            m_AntiKt6H1Topo[23][0] = -0.283737; m_AntiKt6H1Topo[23][1] = -6.17728; m_AntiKt6H1Topo[23][2] = 17.0107; m_AntiKt6H1Topo[23][3] = -13.6622; 
            m_AntiKt6H1Topo[24][0] = -0.299086; m_AntiKt6H1Topo[24][1] = -6.73421; m_AntiKt6H1Topo[24][2] = 20.4318; m_AntiKt6H1Topo[24][3] = -18.2762; 
            m_AntiKt6H1Topo[25][0] = -0.943061; m_AntiKt6H1Topo[25][1] = -1.21085; m_AntiKt6H1Topo[25][2] = 4.5216; m_AntiKt6H1Topo[25][3] = -2.48307; 
            m_AntiKt6H1Topo[26][0] = -0.48727; m_AntiKt6H1Topo[26][1] = -5.07157; m_AntiKt6H1Topo[26][2] = 16.0091; m_AntiKt6H1Topo[26][3] = -13.8729; 
            m_AntiKt6H1Topo[27][0] = -0.911438; m_AntiKt6H1Topo[27][1] = -1.28576; m_AntiKt6H1Topo[27][2] = 4.47514; m_AntiKt6H1Topo[27][3] = -2.35127; 
            m_AntiKt6H1Topo[28][0] = -0.626118; m_AntiKt6H1Topo[28][1] = -4.21669; m_AntiKt6H1Topo[28][2] = 13.4081;  m_AntiKt6H1Topo[28][3] = -11.3929; 
            m_AntiKt6H1Topo[29][0] = -0.963458; m_AntiKt6H1Topo[29][1] = -1.94427; m_AntiKt6H1Topo[29][2] = 7.33021; m_AntiKt6H1Topo[29][3] = -5.96605; 
            m_AntiKt6H1Topo[30][0] = -1.13151; m_AntiKt6H1Topo[30][1] = -1.56778; m_AntiKt6H1Topo[30][2] = 6.78688; m_AntiKt6H1Topo[30][3] = -5.23567; 
            m_AntiKt6H1Topo[31][0] = -1.57391; m_AntiKt6H1Topo[31][1] = -1.7708; m_AntiKt6H1Topo[31][2] = 13.467; m_AntiKt6H1Topo[31][3] = -14.828; 
            m_AntiKt6H1Topo[32][0] = -3.39539; m_AntiKt6H1Topo[32][1] = 9.70605; m_AntiKt6H1Topo[32][2] = -11.8343; m_AntiKt6H1Topo[32][3] = 4.5743; 
            m_AntiKt6H1Topo[33][0] = -3.95261; m_AntiKt6H1Topo[33][1] = 17.5622; m_AntiKt6H1Topo[33][2] = -39.5824; m_AntiKt6H1Topo[33][3] = 34.3734; 
            m_AntiKt6H1Topo[34][0] = 0.0318764; m_AntiKt6H1Topo[34][1] = -10.1119; m_AntiKt6H1Topo[34][2] = 29.3263; m_AntiKt6H1Topo[34][3] = -25.2184; 
            m_AntiKt6H1Topo[35][0] = -0.344382; m_AntiKt6H1Topo[35][1] = -3.12503; m_AntiKt6H1Topo[35][2] = 4.80667; m_AntiKt6H1Topo[35][3] = 0.00960191; 
            m_AntiKt6H1Topo[36][0] = -0.265061; m_AntiKt6H1Topo[36][1] = -4.17192; m_AntiKt6H1Topo[36][2] = 10.5905; m_AntiKt6H1Topo[36][3] = -8.12998; 
            m_AntiKt6H1Topo[37][0] = -0.992244; m_AntiKt6H1Topo[37][1] = 1.66835; m_AntiKt6H1Topo[37][2] = -5.14251; m_AntiKt6H1Topo[37][3] = 6.57914; 
            m_AntiKt6H1Topo[38][0] = -1.4656; m_AntiKt6H1Topo[38][1] = 5.68716; m_AntiKt6H1Topo[38][2] = -17.1531; m_AntiKt6H1Topo[38][3] = 19.0366; 
            m_AntiKt6H1Topo[39][0] = -1.13527; m_AntiKt6H1Topo[39][1] = 1.82556; m_AntiKt6H1Topo[39][2] = -3.67579; m_AntiKt6H1Topo[39][3] = 4.48026; 
            m_AntiKt6H1Topo[40][0] = -2.05429; m_AntiKt6H1Topo[40][1] = 9.43394; m_AntiKt6H1Topo[40][2] = -23.6326; m_AntiKt6H1Topo[40][3] = 21.6411; 
            m_AntiKt6H1Topo[41][0] = -3.18889; m_AntiKt6H1Topo[41][1] = 19.2942; m_AntiKt6H1Topo[41][2] = -50.9616; m_AntiKt6H1Topo[41][3] = 46.7402; 
            m_AntiKt6H1Topo[42][0] = -0.315742; m_AntiKt6H1Topo[42][1] = -4.09849; m_AntiKt6H1Topo[42][2] = 12.3794; m_AntiKt6H1Topo[42][3] = -9.94372; 
            m_AntiKt6H1Topo[43][0] = 0.569235; m_AntiKt6H1Topo[43][1] = -13.3529; m_AntiKt6H1Topo[43][2] = 41.7681; m_AntiKt6H1Topo[43][3] = -39.8164; 
            m_AntiKt6H1Topo[44][0] = -0.568405; m_AntiKt6H1Topo[44][1] = -3.98445; m_AntiKt6H1Topo[44][2] = 15.0592; m_AntiKt6H1Topo[44][3] = -16.0003;



            m_AntiKt4H1Topo[0][0] = -0.676187 ;  m_AntiKt4H1Topo[0][1] = -6.860650 ;  m_AntiKt4H1Topo[0][2] = 23.558500 ;  m_AntiKt4H1Topo[0][3] = -23.062900 ; 
            m_AntiKt4H1Topo[1][0] = -0.629936 ;  m_AntiKt4H1Topo[1][1] = -6.392060 ;  m_AntiKt4H1Topo[1][2] = 20.163100 ;  m_AntiKt4H1Topo[1][3] = -17.983900 ; 
            m_AntiKt4H1Topo[2][0] = -0.557654 ;  m_AntiKt4H1Topo[2][1] = -6.854110 ;  m_AntiKt4H1Topo[2][2] = 21.357100 ;  m_AntiKt4H1Topo[2][3] = -19.077900 ; 
            m_AntiKt4H1Topo[3][0] = -0.520919 ;  m_AntiKt4H1Topo[3][1] = -7.372490 ;  m_AntiKt4H1Topo[3][2] = 23.490000 ;  m_AntiKt4H1Topo[3][3] = -21.711200 ; 
            m_AntiKt4H1Topo[4][0] = -0.477992 ;  m_AntiKt4H1Topo[4][1] = -7.460880 ;  m_AntiKt4H1Topo[4][2] = 22.848700 ;  m_AntiKt4H1Topo[4][3] = -20.338000 ; 
            m_AntiKt4H1Topo[5][0] = -0.522237 ;  m_AntiKt4H1Topo[5][1] = -7.143730 ;  m_AntiKt4H1Topo[5][2] = 21.787200 ;  m_AntiKt4H1Topo[5][3] = -19.072100 ; 
            m_AntiKt4H1Topo[6][0] = -0.308257 ;  m_AntiKt4H1Topo[6][1] = -9.384820 ;  m_AntiKt4H1Topo[6][2] = 28.849100 ;  m_AntiKt4H1Topo[6][3] = -26.528700 ; 
            m_AntiKt4H1Topo[7][0] = -0.497004 ;  m_AntiKt4H1Topo[7][1] = -8.458420 ;  m_AntiKt4H1Topo[7][2] = 26.748600 ;  m_AntiKt4H1Topo[7][3] = -24.513800 ; 
            m_AntiKt4H1Topo[8][0] = -0.807465 ;  m_AntiKt4H1Topo[8][1] = -6.139470 ;  m_AntiKt4H1Topo[8][2] = 20.236700 ;  m_AntiKt4H1Topo[8][3] = -18.237500 ; 
            m_AntiKt4H1Topo[9][0] = -1.161570 ;  m_AntiKt4H1Topo[9][1] = -3.047230 ;  m_AntiKt4H1Topo[9][2] = 10.972900 ;  m_AntiKt4H1Topo[9][3] = -8.901660 ; 
            m_AntiKt4H1Topo[10][0] = -0.074286 ;  m_AntiKt4H1Topo[10][1] = -12.301200 ;  m_AntiKt4H1Topo[10][2] = 38.239700 ;  m_AntiKt4H1Topo[10][3] = -36.071600 ; 
            m_AntiKt4H1Topo[11][0] = 0.321388 ;  m_AntiKt4H1Topo[11][1] = -14.895100 ;  m_AntiKt4H1Topo[11][2] = 43.894900 ;  m_AntiKt4H1Topo[11][3] = -39.907100 ; 
            m_AntiKt4H1Topo[12][0] = 0.216501 ;  m_AntiKt4H1Topo[12][1] = -14.684700 ;  m_AntiKt4H1Topo[12][2] = 45.044000 ;  m_AntiKt4H1Topo[12][3] = -42.762000 ; 
            m_AntiKt4H1Topo[13][0] = -0.477564 ;  m_AntiKt4H1Topo[13][1] = -10.073800 ;  m_AntiKt4H1Topo[13][2] = 33.532300 ;  m_AntiKt4H1Topo[13][3] = -32.385600 ; 
            m_AntiKt4H1Topo[14][0] = -1.511520 ;  m_AntiKt4H1Topo[14][1] = -2.076930 ;  m_AntiKt4H1Topo[14][2] = 12.346600 ;  m_AntiKt4H1Topo[14][3] = -13.477600 ; 
            m_AntiKt4H1Topo[15][0] = -0.674486 ;  m_AntiKt4H1Topo[15][1] = -5.834050 ;  m_AntiKt4H1Topo[15][2] = 16.882600 ;  m_AntiKt4H1Topo[15][3] = -13.340900 ; 
            m_AntiKt4H1Topo[16][0] = -0.610663 ;  m_AntiKt4H1Topo[16][1] = -5.084080 ;  m_AntiKt4H1Topo[16][2] = 13.095700 ;  m_AntiKt4H1Topo[16][3] = -8.524270 ; 
            m_AntiKt4H1Topo[17][0] = -0.223220 ;  m_AntiKt4H1Topo[17][1] = -7.489740 ;  m_AntiKt4H1Topo[17][2] = 19.076100 ;  m_AntiKt4H1Topo[17][3] = -13.758100 ; 
            m_AntiKt4H1Topo[18][0] = -0.188766 ;  m_AntiKt4H1Topo[18][1] = -7.641110 ;  m_AntiKt4H1Topo[18][2] = 20.650900 ;  m_AntiKt4H1Topo[18][3] = -16.644800 ; 
            m_AntiKt4H1Topo[19][0] = -0.288441 ;  m_AntiKt4H1Topo[19][1] = -6.262240 ;  m_AntiKt4H1Topo[19][2] = 15.924400 ;  m_AntiKt4H1Topo[19][3] = -11.327000 ; 
            m_AntiKt4H1Topo[20][0] = -0.421009 ;  m_AntiKt4H1Topo[20][1] = -5.122200 ;  m_AntiKt4H1Topo[20][2] = 13.082800 ;  m_AntiKt4H1Topo[20][3] = -8.952590 ; 
            m_AntiKt4H1Topo[21][0] = -0.414494 ;  m_AntiKt4H1Topo[21][1] = -5.026530 ;  m_AntiKt4H1Topo[21][2] = 12.662200 ;  m_AntiKt4H1Topo[21][3] = -8.284470 ; 
            m_AntiKt4H1Topo[22][0] = -0.317653 ;  m_AntiKt4H1Topo[22][1] = -6.132450 ;  m_AntiKt4H1Topo[22][2] = 17.535000 ;  m_AntiKt4H1Topo[22][3] = -14.749800 ; 
            m_AntiKt4H1Topo[23][0] = -0.218058 ;  m_AntiKt4H1Topo[23][1] = -7.142160 ;  m_AntiKt4H1Topo[23][2] = 21.112800 ;  m_AntiKt4H1Topo[23][3] = -18.402800 ; 
            m_AntiKt4H1Topo[24][0] = -0.331280 ;  m_AntiKt4H1Topo[24][1] = -6.530000 ;  m_AntiKt4H1Topo[24][2] = 19.926800 ;  m_AntiKt4H1Topo[24][3] = -17.420500 ; 
            m_AntiKt4H1Topo[25][0] = -0.949445 ;  m_AntiKt4H1Topo[25][1] = -1.804750 ;  m_AntiKt4H1Topo[25][2] = 7.980680 ;  m_AntiKt4H1Topo[25][3] = -7.158580 ; 
            m_AntiKt4H1Topo[26][0] = -0.755198 ;  m_AntiKt4H1Topo[26][1] = -2.699290 ;  m_AntiKt4H1Topo[26][2] = 9.033340 ;  m_AntiKt4H1Topo[26][3] = -6.903530 ; 
            m_AntiKt4H1Topo[27][0] = -1.053270 ;  m_AntiKt4H1Topo[27][1] = -0.555422 ;  m_AntiKt4H1Topo[27][2] = 4.031690 ;  m_AntiKt4H1Topo[27][3] = -3.051090 ; 
            m_AntiKt4H1Topo[28][0] = -0.681965 ;  m_AntiKt4H1Topo[28][1] = -4.626860 ;  m_AntiKt4H1Topo[28][2] = 17.521700 ;  m_AntiKt4H1Topo[28][3] = -17.399100 ; 
            m_AntiKt4H1Topo[29][0] = -1.085640 ;  m_AntiKt4H1Topo[29][1] = -1.463310 ;  m_AntiKt4H1Topo[29][2] = 7.380520 ;  m_AntiKt4H1Topo[29][3] = -6.701970 ; 
            m_AntiKt4H1Topo[30][0] = -1.606570 ;  m_AntiKt4H1Topo[30][1] = 2.403720 ;  m_AntiKt4H1Topo[30][2] = -5.305410 ;  m_AntiKt4H1Topo[30][3] = 7.094540 ; 
            m_AntiKt4H1Topo[31][0] = -2.220930 ;  m_AntiKt4H1Topo[31][1] = 3.548040 ;  m_AntiKt4H1Topo[31][2] = -3.272110 ;  m_AntiKt4H1Topo[31][3] = 2.328850 ; 
            m_AntiKt4H1Topo[32][0] = -3.459130 ;  m_AntiKt4H1Topo[32][1] = 8.706640 ;  m_AntiKt4H1Topo[32][2] = -8.236530 ;  m_AntiKt4H1Topo[32][3] = 0.544772 ; 
            m_AntiKt4H1Topo[33][0] = -4.327500 ;  m_AntiKt4H1Topo[33][1] = 17.614800 ;  m_AntiKt4H1Topo[33][2] = -35.419800 ;  m_AntiKt4H1Topo[33][3] = 27.574600 ; 
            m_AntiKt4H1Topo[34][0] = 0.664762 ;  m_AntiKt4H1Topo[34][1] = -17.605000 ;  m_AntiKt4H1Topo[34][2] = 54.130200 ;  m_AntiKt4H1Topo[34][3] = -51.190100 ; 
            m_AntiKt4H1Topo[35][0] = -1.240060 ;  m_AntiKt4H1Topo[35][1] = 3.811730 ;  m_AntiKt4H1Topo[35][2] = -13.684600 ;  m_AntiKt4H1Topo[35][3] = 16.516200 ; 
            m_AntiKt4H1Topo[36][0] = -1.103360 ;  m_AntiKt4H1Topo[36][1] = 2.044190 ;  m_AntiKt4H1Topo[36][2] = -5.630570 ;  m_AntiKt4H1Topo[36][3] = 6.593320 ; 
            m_AntiKt4H1Topo[37][0] = -0.803584 ;  m_AntiKt4H1Topo[37][1] = -1.336720 ;  m_AntiKt4H1Topo[37][2] = 5.392730 ;  m_AntiKt4H1Topo[37][3] = -4.371850 ; 
            m_AntiKt4H1Topo[38][0] = -1.465460 ;  m_AntiKt4H1Topo[38][1] = 3.292650 ;  m_AntiKt4H1Topo[38][2] = -5.964320 ;  m_AntiKt4H1Topo[38][3] = 5.321690 ; 
            m_AntiKt4H1Topo[39][0] = -1.978400 ;  m_AntiKt4H1Topo[39][1] = 6.837860 ;  m_AntiKt4H1Topo[39][2] = -14.398100 ;  m_AntiKt4H1Topo[39][3] = 12.091000 ; 
            m_AntiKt4H1Topo[40][0] = -0.766573 ;  m_AntiKt4H1Topo[40][1] = -3.536390 ;  m_AntiKt4H1Topo[40][2] = 15.589000 ;  m_AntiKt4H1Topo[40][3] = -16.967200 ; 
            m_AntiKt4H1Topo[41][0] = -2.075530 ;  m_AntiKt4H1Topo[41][1] = 9.745690 ;  m_AntiKt4H1Topo[41][2] = -26.352200 ;  m_AntiKt4H1Topo[41][3] = 26.050700 ; 
            m_AntiKt4H1Topo[42][0] = 3.867300 ;  m_AntiKt4H1Topo[42][1] = -41.041400 ;  m_AntiKt4H1Topo[42][2] = 118.577000 ;  m_AntiKt4H1Topo[42][3] = -110.930000 ; 
            m_AntiKt4H1Topo[43][0] = -0.093523 ;  m_AntiKt4H1Topo[43][1] = -7.380870 ;  m_AntiKt4H1Topo[43][2] = 22.895800 ;  m_AntiKt4H1Topo[43][3] = -19.853600 ; 
            m_AntiKt4H1Topo[44][0] = 0.453499 ;  m_AntiKt4H1Topo[44][1] = -11.350800 ;  m_AntiKt4H1Topo[44][2] = 31.127900 ;  m_AntiKt4H1Topo[44][3] = -26.921500 ; 


            m_AntiKt6H1Tower[0][0] = -0.531828 ;  m_AntiKt6H1Tower[0][1] = -8.815712 ;  m_AntiKt6H1Tower[0][2] = 31.487768 ;  m_AntiKt6H1Tower[0][3] = -32.557412 ; 
            m_AntiKt6H1Tower[1][0] = -0.595853 ;  m_AntiKt6H1Tower[1][1] = -7.036562 ;  m_AntiKt6H1Tower[1][2] = 22.839277 ;  m_AntiKt6H1Tower[1][3] = -20.613444 ; 
            m_AntiKt6H1Tower[2][0] = -0.400570 ;  m_AntiKt6H1Tower[2][1] = -9.293366 ;  m_AntiKt6H1Tower[2][2] = 32.309430 ;  m_AntiKt6H1Tower[2][3] = -33.899515 ; 
            m_AntiKt6H1Tower[3][0] = -0.403523 ;  m_AntiKt6H1Tower[3][1] = -9.096110 ;  m_AntiKt6H1Tower[3][2] = 30.621969 ;  m_AntiKt6H1Tower[3][3] = -30.395132 ; 
            m_AntiKt6H1Tower[4][0] = -0.431400 ;  m_AntiKt6H1Tower[4][1] = -8.228697 ;  m_AntiKt6H1Tower[4][2] = 25.630379 ;  m_AntiKt6H1Tower[4][3] = -22.736815 ; 
            m_AntiKt6H1Tower[5][0] = -0.405739 ;  m_AntiKt6H1Tower[5][1] = -8.864647 ;  m_AntiKt6H1Tower[5][2] = 28.762347 ;  m_AntiKt6H1Tower[5][3] = -27.535987 ; 
            m_AntiKt6H1Tower[6][0] = -0.174485 ;  m_AntiKt6H1Tower[6][1] = -11.327571 ;  m_AntiKt6H1Tower[6][2] = 36.786964 ;  m_AntiKt6H1Tower[6][3] = -36.221364 ; 
            m_AntiKt6H1Tower[7][0] = -0.202916 ;  m_AntiKt6H1Tower[7][1] = -12.499841 ;  m_AntiKt6H1Tower[7][2] = 43.755044 ;  m_AntiKt6H1Tower[7][3] = -46.580174 ; 
            m_AntiKt6H1Tower[8][0] = -0.608770 ;  m_AntiKt6H1Tower[8][1] = -8.973535 ;  m_AntiKt6H1Tower[8][2] = 32.230743 ;  m_AntiKt6H1Tower[8][3] = -33.642838 ; 
            m_AntiKt6H1Tower[9][0] = -1.173041 ;  m_AntiKt6H1Tower[9][1] = -3.014877 ;  m_AntiKt6H1Tower[9][2] = 10.817822 ;  m_AntiKt6H1Tower[9][3] = -8.090800 ; 
            m_AntiKt6H1Tower[10][0] = 0.086048 ;  m_AntiKt6H1Tower[10][1] = -14.586995 ;  m_AntiKt6H1Tower[10][2] = 47.649213 ;  m_AntiKt6H1Tower[10][3] = -48.020763 ; 
            m_AntiKt6H1Tower[11][0] = 0.754505 ;  m_AntiKt6H1Tower[11][1] = -20.817996 ;  m_AntiKt6H1Tower[11][2] = 68.508386 ;  m_AntiKt6H1Tower[11][3] = -72.360015 ; 
            m_AntiKt6H1Tower[12][0] = 0.490883 ;  m_AntiKt6H1Tower[12][1] = -18.478448 ;  m_AntiKt6H1Tower[12][2] = 60.673390 ;  m_AntiKt6H1Tower[12][3] = -62.912017 ; 
            m_AntiKt6H1Tower[13][0] = -0.503989 ;  m_AntiKt6H1Tower[13][1] = -10.147999 ;  m_AntiKt6H1Tower[13][2] = 35.088032 ;  m_AntiKt6H1Tower[13][3] = -35.187702 ; 
            m_AntiKt6H1Tower[14][0] = -1.607949 ;  m_AntiKt6H1Tower[14][1] = -0.939133 ;  m_AntiKt6H1Tower[14][2] = 8.469567 ;  m_AntiKt6H1Tower[14][3] = -8.629370 ; 
            m_AntiKt6H1Tower[15][0] = -0.860203 ;  m_AntiKt6H1Tower[15][1] = -3.322783 ;  m_AntiKt6H1Tower[15][2] = 6.061165 ;  m_AntiKt6H1Tower[15][3] = 2.006822 ; 
            m_AntiKt6H1Tower[16][0] = -0.746905 ;  m_AntiKt6H1Tower[16][1] = -3.491616 ;  m_AntiKt6H1Tower[16][2] = 7.012125 ;  m_AntiKt6H1Tower[16][3] = -0.481971 ; 
            m_AntiKt6H1Tower[17][0] = -0.296259 ;  m_AntiKt6H1Tower[17][1] = -6.672796 ;  m_AntiKt6H1Tower[17][2] = 16.228749 ;  m_AntiKt6H1Tower[17][3] = -10.169059 ; 
            m_AntiKt6H1Tower[18][0] = -0.076723 ;  m_AntiKt6H1Tower[18][1] = -9.206198 ;  m_AntiKt6H1Tower[18][2] = 27.446328 ;  m_AntiKt6H1Tower[18][3] = -25.647870 ; 
            m_AntiKt6H1Tower[19][0] = -0.174771 ;  m_AntiKt6H1Tower[19][1] = -7.380293 ;  m_AntiKt6H1Tower[19][2] = 19.244811 ;  m_AntiKt6H1Tower[19][3] = -14.141259 ; 
            m_AntiKt6H1Tower[20][0] = -0.469554 ;  m_AntiKt6H1Tower[20][1] = -4.401218 ;  m_AntiKt6H1Tower[20][2] = 9.633196 ;  m_AntiKt6H1Tower[20][3] = -3.667523 ; 
            m_AntiKt6H1Tower[21][0] = -0.404119 ;  m_AntiKt6H1Tower[21][1] = -4.996836 ;  m_AntiKt6H1Tower[21][2] = 11.953167 ;  m_AntiKt6H1Tower[21][3] = -6.490091 ; 
            m_AntiKt6H1Tower[22][0] = -0.253114 ;  m_AntiKt6H1Tower[22][1] = -6.669808 ;  m_AntiKt6H1Tower[22][2] = 18.474525 ;  m_AntiKt6H1Tower[22][3] = -14.474722 ; 
            m_AntiKt6H1Tower[23][0] = 0.045939 ;  m_AntiKt6H1Tower[23][1] = -9.890581 ;  m_AntiKt6H1Tower[23][2] = 30.125349 ;  m_AntiKt6H1Tower[23][3] = -28.039000 ; 
            m_AntiKt6H1Tower[24][0] = -0.374858 ;  m_AntiKt6H1Tower[24][1] = -6.554773 ;  m_AntiKt6H1Tower[24][2] = 20.981801 ;  m_AntiKt6H1Tower[24][3] = -19.337984 ; 
            m_AntiKt6H1Tower[25][0] = -1.025748 ;  m_AntiKt6H1Tower[25][1] = -0.164423 ;  m_AntiKt6H1Tower[25][2] = -0.285868 ;  m_AntiKt6H1Tower[25][3] = 4.818927 ; 
            m_AntiKt6H1Tower[26][0] = -0.643836 ;  m_AntiKt6H1Tower[26][1] = -3.584475 ;  m_AntiKt6H1Tower[26][2] = 10.626951 ;  m_AntiKt6H1Tower[26][3] = -6.927664 ; 
            m_AntiKt6H1Tower[27][0] = -1.457040 ;  m_AntiKt6H1Tower[27][1] = 4.538457 ;  m_AntiKt6H1Tower[27][2] = -16.665680 ;  m_AntiKt6H1Tower[27][3] = 23.311161 ; 
            m_AntiKt6H1Tower[28][0] = -0.501388 ;  m_AntiKt6H1Tower[28][1] = -5.785419 ;  m_AntiKt6H1Tower[28][2] = 18.906817 ;  m_AntiKt6H1Tower[28][3] = -17.119481 ; 
            m_AntiKt6H1Tower[29][0] = -0.480514 ;  m_AntiKt6H1Tower[29][1] = -7.494461 ;  m_AntiKt6H1Tower[29][2] = 27.242735 ;  m_AntiKt6H1Tower[29][3] = -28.825547 ; 
            m_AntiKt6H1Tower[30][0] = -0.767416 ;  m_AntiKt6H1Tower[30][1] = -5.907525 ;  m_AntiKt6H1Tower[30][2] = 22.526548 ;  m_AntiKt6H1Tower[30][3] = -23.372494 ; 
            m_AntiKt6H1Tower[31][0] = -1.888214 ;  m_AntiKt6H1Tower[31][1] = 0.737430 ;  m_AntiKt6H1Tower[31][2] = 5.876736 ;  m_AntiKt6H1Tower[31][3] = -6.281703 ; 
            m_AntiKt6H1Tower[32][0] = -2.273965 ;  m_AntiKt6H1Tower[32][1] = -1.377201 ;  m_AntiKt6H1Tower[32][2] = 23.734253 ;  m_AntiKt6H1Tower[32][3] = -32.886162 ; 
            m_AntiKt6H1Tower[33][0] = -2.860770 ;  m_AntiKt6H1Tower[33][1] = 5.414834 ;  m_AntiKt6H1Tower[33][2] = 3.191877 ;  m_AntiKt6H1Tower[33][3] = -14.216696 ; 
            m_AntiKt6H1Tower[34][0] = -0.320954 ;  m_AntiKt6H1Tower[34][1] = -6.924036 ;  m_AntiKt6H1Tower[34][2] = 19.316472 ;  m_AntiKt6H1Tower[34][3] = -14.439498 ; 
            m_AntiKt6H1Tower[35][0] = 0.273270 ;  m_AntiKt6H1Tower[35][1] = -9.936829 ;  m_AntiKt6H1Tower[35][2] = 28.442339 ;  m_AntiKt6H1Tower[35][3] = -26.048901 ; 
            m_AntiKt6H1Tower[36][0] = -0.620211 ;  m_AntiKt6H1Tower[36][1] = -1.472639 ;  m_AntiKt6H1Tower[36][2] = 3.222407 ;  m_AntiKt6H1Tower[36][3] = -0.827148 ; 
            m_AntiKt6H1Tower[37][0] = -1.261469 ;  m_AntiKt6H1Tower[37][1] = 4.795849 ;  m_AntiKt6H1Tower[37][2] = -17.569881 ;  m_AntiKt6H1Tower[37][3] = 22.471654 ; 
            m_AntiKt6H1Tower[38][0] = -3.958829 ;  m_AntiKt6H1Tower[38][1] = 30.459187 ;  m_AntiKt6H1Tower[38][2] = -99.290640 ;  m_AntiKt6H1Tower[38][3] = 109.125282 ; 
            m_AntiKt6H1Tower[39][0] = -2.618143 ;  m_AntiKt6H1Tower[39][1] = 14.230753 ;  m_AntiKt6H1Tower[39][2] = -38.545348 ;  m_AntiKt6H1Tower[39][3] = 36.995610 ; 
            m_AntiKt6H1Tower[40][0] = -2.535882 ;  m_AntiKt6H1Tower[40][1] = 14.176748 ;  m_AntiKt6H1Tower[40][2] = -40.452949 ;  m_AntiKt6H1Tower[40][3] = 41.603819 ; 
            m_AntiKt6H1Tower[41][0] = -5.684856 ;  m_AntiKt6H1Tower[41][1] = 42.549219 ;  m_AntiKt6H1Tower[41][2] = -124.054841 ;  m_AntiKt6H1Tower[41][3] = 122.994901 ; 
            m_AntiKt6H1Tower[42][0] = 0.744114 ;  m_AntiKt6H1Tower[42][1] = -12.967402 ;  m_AntiKt6H1Tower[42][2] = 35.420261 ;  m_AntiKt6H1Tower[42][3] = -28.954084 ; 
            m_AntiKt6H1Tower[43][0] = 7.224882 ;  m_AntiKt6H1Tower[43][1] = -75.228986 ;  m_AntiKt6H1Tower[43][2] = 232.034496 ;  m_AntiKt6H1Tower[43][3] = -233.827046 ; 
            m_AntiKt6H1Tower[44][0] = 0.887661 ;  m_AntiKt6H1Tower[44][1] = -14.836962 ;  m_AntiKt6H1Tower[44][2] = 40.392890 ;  m_AntiKt6H1Tower[44][3] = -34.100683 ; 


            m_AntiKt4H1Tower[0][0] = -0.646831 ;  m_AntiKt4H1Tower[0][1] = -7.148533 ;  m_AntiKt4H1Tower[0][2] = 24.029264 ;  m_AntiKt4H1Tower[0][3] = -21.928339 ; 
            m_AntiKt4H1Tower[1][0] = -0.536843 ;  m_AntiKt4H1Tower[1][1] = -7.665576 ;  m_AntiKt4H1Tower[1][2] = 25.409125 ;  m_AntiKt4H1Tower[1][3] = -24.037261 ; 
            m_AntiKt4H1Tower[2][0] = -0.453207 ;  m_AntiKt4H1Tower[2][1] = -8.380156 ;  m_AntiKt4H1Tower[2][2] = 28.037993 ;  m_AntiKt4H1Tower[2][3] = -27.499049 ; 
            m_AntiKt4H1Tower[3][0] = -0.348502 ;  m_AntiKt4H1Tower[3][1] = -9.713273 ;  m_AntiKt4H1Tower[3][2] = 33.323519 ;  m_AntiKt4H1Tower[3][3] = -34.129571 ; 
            m_AntiKt4H1Tower[4][0] = -0.284032 ;  m_AntiKt4H1Tower[4][1] = -10.044346 ;  m_AntiKt4H1Tower[4][2] = 33.449909 ;  m_AntiKt4H1Tower[4][3] = -33.475001 ; 
            m_AntiKt4H1Tower[5][0] = -0.300257 ;  m_AntiKt4H1Tower[5][1] = -10.127966 ;  m_AntiKt4H1Tower[5][2] = 34.122312 ;  m_AntiKt4H1Tower[5][3] = -34.617882 ; 
            m_AntiKt4H1Tower[6][0] = -0.272407 ;  m_AntiKt4H1Tower[6][1] = -9.869314 ;  m_AntiKt4H1Tower[6][2] = 30.359042 ;  m_AntiKt4H1Tower[6][3] = -27.000385 ; 
            m_AntiKt4H1Tower[7][0] = -0.166548 ;  m_AntiKt4H1Tower[7][1] = -12.872234 ;  m_AntiKt4H1Tower[7][2] = 44.931068 ;  m_AntiKt4H1Tower[7][3] = -47.793719 ; 
            m_AntiKt4H1Tower[8][0] = -0.540306 ;  m_AntiKt4H1Tower[8][1] = -9.804960 ;  m_AntiKt4H1Tower[8][2] = 35.362786 ;  m_AntiKt4H1Tower[8][3] = -37.472033 ; 
            m_AntiKt4H1Tower[9][0] = -1.296938 ;  m_AntiKt4H1Tower[9][1] = -1.574760 ;  m_AntiKt4H1Tower[9][2] = 4.755894 ;  m_AntiKt4H1Tower[9][3] = 0.509835 ; 
            m_AntiKt4H1Tower[10][0] = 0.019177 ;  m_AntiKt4H1Tower[10][1] = -13.457964 ;  m_AntiKt4H1Tower[10][2] = 41.778874 ;  m_AntiKt4H1Tower[10][3] = -38.445683 ; 
            m_AntiKt4H1Tower[11][0] = 0.351161 ;  m_AntiKt4H1Tower[11][1] = -15.396211 ;  m_AntiKt4H1Tower[11][2] = 45.279627 ;  m_AntiKt4H1Tower[11][3] = -40.196276 ; 
            m_AntiKt4H1Tower[12][0] = 0.681338 ;  m_AntiKt4H1Tower[12][1] = -20.774074 ;  m_AntiKt4H1Tower[12][2] = 69.307446 ;  m_AntiKt4H1Tower[12][3] = -73.187152 ; 
            m_AntiKt4H1Tower[13][0] = -0.104148 ;  m_AntiKt4H1Tower[13][1] = -15.115494 ;  m_AntiKt4H1Tower[13][2] = 54.030780 ;  m_AntiKt4H1Tower[13][3] = -58.479533 ; 
            m_AntiKt4H1Tower[14][0] = -1.573951 ;  m_AntiKt4H1Tower[14][1] = -1.726078 ;  m_AntiKt4H1Tower[14][2] = 11.025080 ;  m_AntiKt4H1Tower[14][3] = -10.755109 ; 
            m_AntiKt4H1Tower[15][0] = -0.609762 ;  m_AntiKt4H1Tower[15][1] = -6.433901 ;  m_AntiKt4H1Tower[15][2] = 18.197556 ;  m_AntiKt4H1Tower[15][3] = -13.134154 ; 
            m_AntiKt4H1Tower[16][0] = -0.484372 ;  m_AntiKt4H1Tower[16][1] = -6.837282 ;  m_AntiKt4H1Tower[16][2] = 19.908900 ;  m_AntiKt4H1Tower[16][3] = -15.868226 ; 
            m_AntiKt4H1Tower[17][0] = 0.044881 ;  m_AntiKt4H1Tower[17][1] = -11.009338 ;  m_AntiKt4H1Tower[17][2] = 33.650955 ;  m_AntiKt4H1Tower[17][3] = -32.157347 ; 
            m_AntiKt4H1Tower[18][0] = -0.168352 ;  m_AntiKt4H1Tower[18][1] = -8.073527 ;  m_AntiKt4H1Tower[18][2] = 22.648551 ;  m_AntiKt4H1Tower[18][3] = -18.385273 ; 
            m_AntiKt4H1Tower[19][0] = -0.195094 ;  m_AntiKt4H1Tower[19][1] = -7.440244 ;  m_AntiKt4H1Tower[19][2] = 20.326958 ;  m_AntiKt4H1Tower[19][3] = -15.630561 ; 
            m_AntiKt4H1Tower[20][0] = -0.638136 ;  m_AntiKt4H1Tower[20][1] = -2.753472 ;  m_AntiKt4H1Tower[20][2] = 4.274174 ;  m_AntiKt4H1Tower[20][3] = 2.649932 ; 
            m_AntiKt4H1Tower[21][0] = -0.559535 ;  m_AntiKt4H1Tower[21][1] = -3.460478 ;  m_AntiKt4H1Tower[21][2] = 6.488394 ;  m_AntiKt4H1Tower[21][3] = 0.780263 ; 
            m_AntiKt4H1Tower[22][0] = -0.124555 ;  m_AntiKt4H1Tower[22][1] = -8.703811 ;  m_AntiKt4H1Tower[22][2] = 27.244515 ;  m_AntiKt4H1Tower[22][3] = -25.757030 ; 
            m_AntiKt4H1Tower[23][0] = -0.240710 ;  m_AntiKt4H1Tower[23][1] = -7.018678 ;  m_AntiKt4H1Tower[23][2] = 19.515591 ;  m_AntiKt4H1Tower[23][3] = -14.419309 ; 
            m_AntiKt4H1Tower[24][0] = -0.142195 ;  m_AntiKt4H1Tower[24][1] = -9.627801 ;  m_AntiKt4H1Tower[24][2] = 32.050141 ;  m_AntiKt4H1Tower[24][3] = -31.683310 ; 
            m_AntiKt4H1Tower[25][0] = -1.221964 ;  m_AntiKt4H1Tower[25][1] = 1.137163 ;  m_AntiKt4H1Tower[25][2] = -4.316276 ;  m_AntiKt4H1Tower[25][3] = 9.693247 ; 
            m_AntiKt4H1Tower[26][0] = -0.648855 ;  m_AntiKt4H1Tower[26][1] = -4.349432 ;  m_AntiKt4H1Tower[26][2] = 13.845221 ;  m_AntiKt4H1Tower[26][3] = -10.509775 ; 
            m_AntiKt4H1Tower[27][0] = -0.944168 ;  m_AntiKt4H1Tower[27][1] = -2.338734 ;  m_AntiKt4H1Tower[27][2] = 9.758786 ;  m_AntiKt4H1Tower[27][3] = -8.375663 ; 
            m_AntiKt4H1Tower[28][0] = -0.693765 ;  m_AntiKt4H1Tower[28][1] = -5.296733 ;  m_AntiKt4H1Tower[28][2] = 19.918827 ;  m_AntiKt4H1Tower[28][3] = -19.506838 ; 
            m_AntiKt4H1Tower[29][0] = -0.782081 ;  m_AntiKt4H1Tower[29][1] = -5.263714 ;  m_AntiKt4H1Tower[29][2] = 19.931353 ;  m_AntiKt4H1Tower[29][3] = -19.448692 ; 
            m_AntiKt4H1Tower[30][0] = -2.192686 ;  m_AntiKt4H1Tower[30][1] = 7.326015 ;  m_AntiKt4H1Tower[30][2] = -20.233413 ;  m_AntiKt4H1Tower[30][3] = 22.794518 ; 
            m_AntiKt4H1Tower[31][0] = -2.490445 ;  m_AntiKt4H1Tower[31][1] = 7.874842 ;  m_AntiKt4H1Tower[31][2] = -24.556146 ;  m_AntiKt4H1Tower[31][3] = 33.887220 ; 
            m_AntiKt4H1Tower[32][0] = -9.098189 ;  m_AntiKt4H1Tower[32][1] = 69.314193 ;  m_AntiKt4H1Tower[32][2] = -223.784830 ;  m_AntiKt4H1Tower[32][3] = 253.401620 ; 
            m_AntiKt4H1Tower[33][0] = -7.948353 ;  m_AntiKt4H1Tower[33][1] = 53.803841 ;  m_AntiKt4H1Tower[33][2] = -155.977264 ;  m_AntiKt4H1Tower[33][3] = 161.448727 ; 
            m_AntiKt4H1Tower[34][0] = -0.593530 ;  m_AntiKt4H1Tower[34][1] = -5.847806 ;  m_AntiKt4H1Tower[34][2] = 17.641616 ;  m_AntiKt4H1Tower[34][3] = -13.721947 ; 
            m_AntiKt4H1Tower[35][0] = -1.783394 ;  m_AntiKt4H1Tower[35][1] = 8.891799 ;  m_AntiKt4H1Tower[35][2] = -30.533943 ;  m_AntiKt4H1Tower[35][3] = 35.928578 ; 
            m_AntiKt4H1Tower[36][0] = -0.727803 ;  m_AntiKt4H1Tower[36][1] = -0.984880 ;  m_AntiKt4H1Tower[36][2] = 0.534182 ;  m_AntiKt4H1Tower[36][3] = 4.409238 ; 
            m_AntiKt4H1Tower[37][0] = -2.005239 ;  m_AntiKt4H1Tower[37][1] = 9.625854 ;  m_AntiKt4H1Tower[37][2] = -29.403580 ;  m_AntiKt4H1Tower[37][3] = 32.727437 ; 
            m_AntiKt4H1Tower[38][0] = -2.824277 ;  m_AntiKt4H1Tower[38][1] = 16.510914 ;  m_AntiKt4H1Tower[38][2] = -49.574617 ;  m_AntiKt4H1Tower[38][3] = 52.304695 ; 
            m_AntiKt4H1Tower[39][0] = -4.706911 ;  m_AntiKt4H1Tower[39][1] = 30.872244 ;  m_AntiKt4H1Tower[39][2] = -86.584461 ;  m_AntiKt4H1Tower[39][3] = 84.183582 ; 
            m_AntiKt4H1Tower[40][0] = 0.014897 ;  m_AntiKt4H1Tower[40][1] = -12.454598 ;  m_AntiKt4H1Tower[40][2] = 44.930971 ;  m_AntiKt4H1Tower[40][3] = -47.603190 ; 
            m_AntiKt4H1Tower[41][0] = 0.438302 ;  m_AntiKt4H1Tower[41][1] = -16.107488 ;  m_AntiKt4H1Tower[41][2] = 55.928006 ;  m_AntiKt4H1Tower[41][3] = -58.257899 ; 
            m_AntiKt4H1Tower[42][0] = 5.558652 ;  m_AntiKt4H1Tower[42][1] = -57.429996 ;  m_AntiKt4H1Tower[42][2] = 166.930539 ;  m_AntiKt4H1Tower[42][3] = -157.389177 ; 
            m_AntiKt4H1Tower[43][0] = 0.188391 ;  m_AntiKt4H1Tower[43][1] = -14.877299 ;  m_AntiKt4H1Tower[43][2] = 54.814916 ;  m_AntiKt4H1Tower[43][3] = -59.482146 ; 
            m_AntiKt4H1Tower[44][0] = 2.095088 ;  m_AntiKt4H1Tower[44][1] = -27.416361 ;  m_AntiKt4H1Tower[44][2] = 78.245867 ;  m_AntiKt4H1Tower[44][3] = -70.576159 ; 


        }

};
