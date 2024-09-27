#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLorentzVector.h>



//------------------------------------------------------------------------------
// Particle Class
//
class Particle{

        public:
        Particle();
        Particle(double, double, double, double);
        double   pt, eta, phi, E, m, p[4];
        void     p4(double, double, double, double);
        void     print();
        void     setMass(double);
        double   sintheta();
};

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle(){
        pt = eta = phi = E = m = 0.0;
        p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double _pt, double _eta, double _phi, double _E){
  pt = _pt;
  eta = _eta;
  phi = _phi;
  E = _E;
}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta(){

  double r = pt/sqrt(pt*pt + p[2]*p[2]);
  return(r);
}

void Particle::p4(double pT, double eta, double phi, double energy){

  p[0] = pt*cos(phi);
  p[1] = pt*sin(phi);
  p[2] = pt * sinh(eta);
  p[3] = E;

}

void Particle::setMass(double mass)
{
  m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print(){
        std::cout << std::endl;
        std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << std::endl;
        std::cout << "Mass: " << m << std::endl;
}

//
//*** Lepton daughter class ----------------------------------------------------
//
class Lepton: public Particle{
public:
  Lepton(double _pt, double _eta, double _phi, double _E) : Particle(_pt, _eta, _phi, _E) {
  }
  int     charge;
  void    print();
  void    Set_charge(int input){
    charge = input;
  }
};

void Lepton::print(){
  std::cout << "Lepton" << std::endl;
  std::cout << "4-vector: (" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  sin(theta) = " <<  sintheta() << std::endl;
  std::cout << "Charge: " << charge << std::endl;
  std::cout << "Mass: " << m << std::endl;
  std::cout << std::endl;
}

//
//*** Jet daughter class -------------------------------------------------------
//
class Jet: public Particle{
public:
  Jet(double _pt, double _eta, double _phi, double _E) : Particle(_pt, _eta, _phi, _E) {
  }
  int      flavor;
  void     print();
  void     Set_flavor(int input){
    flavor = input;
  }
};

void Jet::print(){
  std::cout << "Jet" << std::endl;
  std::cout << "4-vector: (" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  sin(theta) = " <<  sintheta() << std::endl;
  std::cout << "Flavor: " << flavor << std::endl;
  std::cout << "Mass: " << m << std::endl;
  std::cout << std::endl;
}

int main() {

        /* ************* */
        /* Input Tree   */
        /* ************* */

        TFile *f      = new TFile("input.root","READ");
        TTree *t1 = (TTree*)(f->Get("t1"));

        // Read the variables from the ROOT tree branches
        t1->SetBranchAddress("lepPt",&lepPt);
        t1->SetBranchAddress("lepEta",&lepEta);
        t1->SetBranchAddress("lepPhi",&lepPhi);
        t1->SetBranchAddress("lepE",&lepE);
        t1->SetBranchAddress("lepQ",&lepQ);

        t1->SetBranchAddress("njets",&njets);
        t1->SetBranchAddress("jetPt",&jetPt);
        t1->SetBranchAddress("jetEta",&jetEta);
        t1->SetBranchAddress("jetPhi",&jetPhi);
        t1->SetBranchAddress("jetE", &jetE);
        t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

        // Total number of events in ROOT tree
        Long64_t nentries = t1->GetEntries();

        for (Long64_t jentry=0; jentry<100;jentry++)
        {
                t1->GetEntry(jentry);
                std::cout<<" Event "<< jentry <<std::endl;

                // loop through leptons
                // only 2 leptons per event in this file
                for (int i=0; i < 2; i++){
                  Lepton lepton(lepPt[i], lepEta[i], lepPhi[i], lepE[i]);
                  lepton.p4(lepPt[i], lepEta[i], lepPhi[i], lepE[i]);
                  lepton.Set_charge(lepQ[i]);
                  double mass2 = (lepton.p[3] * lepton.p[3]) - (lepton.p[0] * lepton.p[0] + lepton.p[1] * lepton.p[1] + lepton.p[2] * lepton.p[2]);
                  lepton.setMass(sqrt(mass2));
                  lepton.print();
                }

                //loop through jets
                for (int i=0; i< njets; i++){
                  Jet jet(jetPt[i], jetEta[i], jetPhi[i], jetE[i]);
                  jet.p4(jetPt[i], jetEta[i], jetPhi[i], jetE[i]);
                  jet.Set_flavor(jetHadronFlavour[i]);
                  double mass2 = (jet.p[3] * jet.p[3]) - (jet.p[0] * jet.p[0] + jet.p[1] * jet.p[1] + jet.p[2] * jet.p[2]);
                  jet.setMass(sqrt(mass2));
                  jet.print();
                }



        } // Loop over all events

        return 0;
} 