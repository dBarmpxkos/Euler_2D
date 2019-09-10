/* This program is designed to simulate the Euler equations
 in 2 dimensions, using the SLIC solver.
 
 It provides output in ASCII format, suitable for plotting in gnuplot.
 
 
 */

#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <limits>

typedef std::array<double,4> State;

enum ConsVar{RHO=0, MOM_X, MOM_Y, ENE};
enum PrimVar{DEN=0, V_X, V_Y, PRE};

// Useful constants
const double Gamma = 1.4; // Adiabatic index of air
const double G8 = Gamma-1;
const double G9 = 1/G8;

// Problem specific data that are needed by all functions:
double finalT;
double minX;
double maxX;
unsigned int cellsX;
double minY;
double maxY;
unsigned int cellsY;

// Convert from conserved variables to primitive variables
State primitive(const State& cons)
{
  State prim;
  
  const double rho = cons[RHO];
  
  double vSquared = 0;
  const double recipRho = 1/rho;
  for(unsigned int d=0 ; d < 2 ; d++)
  {
    prim[V_X + d] = cons[MOM_X+d] * recipRho;
    vSquared += prim[V_X+d]*prim[V_X+d];
  }
  
  const double E = cons[ENE];
  const double p = G8*(E - 0.5*rho*vSquared);
  
  prim[DEN] = rho;
  prim[PRE] = p;
  
  return prim;
}

// Convert primitive state vector to conservative state vector
State conservative(const State& prim)
{
  State cons;
  const double rho = prim[DEN];
  const double p = prim[PRE];
  
  double vSquared = 0;
  for(unsigned int d=0 ; d < 2 ; d++)
  {
    cons[MOM_X + d] = rho*prim[V_X + d];
    vSquared += prim[V_X+d]*prim[V_X+d];
  }
  
  cons[RHO] = rho;
  cons[ENE] = p*G9 + 0.5*rho*vSquared;
  
  return cons;
}

// Compute flux-vector corresponding to given state vector in given coordinate direction
State flux(const State& cons, unsigned int coord)
{
  State prim = primitive(cons);
  
  const double p = prim[PRE];
  const double E = cons[ENE];
  const double momN = cons[MOM_X + coord];
  const double uN = prim[V_X + coord];
  
  State f;
  
  f[RHO] = momN;
  f[ENE] = uN*(E+p);
  
  for(unsigned int d=0 ; d < 2 ; d++)
  {
    f[MOM_X + d] = momN * prim[V_X + d];
  }
  
  f[MOM_X + coord] += p;
  
  return f;
}

// Compute the maximum wave-speed from the given conserved state
double maxSpeed(const State& cons)
{
  const State prim = primitive(cons);
  double p = prim[PRE];
  double a = sqrt(Gamma * p / cons[RHO]);
  double v = std::max(prim[V_X], prim[V_Y]);
  
  return a + fabs(v);
}

// Compute the maximum stable time-step for the given data
double timestep(const std::vector<State>& data, const double dx)
{
  double dt = std::numeric_limits<double>::max();

#pragma omp parallel for collapse(2)
  for(unsigned int j=0 ; j < cellsY ; j++)
  {
    for(unsigned int i=0 ; i < cellsX ; i++)
    {
      double v = maxSpeed(data[i + j*cellsX]);
      double cellDt = dx / v;
      dt = std::min(dt, cellDt);
    }
  }
  return dt;
}

// Compute the FORCE flux between two states uL and uR in coordinate direction coord.
State FORCEflux(const State& uL, const State& uR, double dx, double dt, unsigned int coord)
{
  State fL = flux(uL, coord);
  State fR = flux(uR, coord);
  
  State cellRM;
  for(unsigned int i=0 ; i < 4 ; i++)
  {
    cellRM[i] = 0.5*(uL[i] + uR[i] + dt/dx*(fL[i] - fR[i]));
  }
  
  const State fluxRM = flux(cellRM, coord);
  State force;
  for(unsigned int i=0 ; i < 4 ; i++)
  {
    force[i] = 0.5*(fluxRM[i] + 0.5*(fL[i] + fR[i] + (dx/dt)*(uL[i] - uR[i])));
  }
  return force;
}

// Compute the array of fluxes from the given data array
void computeFluxes(const std::vector<State>& data, std::vector<State>& fluxes, double dx, double dt, unsigned int coord)
{
#pragma omp parallel for collapse(2)
  for(unsigned int j=1 ; j < cellsY ; j++)
  {
    for(unsigned int i=1 ; i < cellsX ; i++)
    {
      unsigned int idx = i + j*cellsX;
      unsigned int idxL = idx - ((coord == 0) ? 1 : cellsX);
      fluxes[idx] = FORCEflux(data[idxL], data[idx], dx, dt, coord);
    }
  }
}

// Add the flux array to the data array for the given coordinate direction
void addFluxes(std::vector<State>& data, const std::vector<State>& fluxes, double dx, double dt, unsigned int coord)
{
#pragma omp parallel for collapse(2)
  for(unsigned int j=1 ; j < cellsY-1 ; j++)
  {
    for(unsigned int i=1 ; i < cellsX-1 ; i++)
    {
      unsigned int idx = i + j*cellsX;
      unsigned int idxR = idx + ((coord == 0) ? 1 : cellsX);
      for(unsigned int v=0 ; v < 4 ; v++)
      {
        data[idx][v] = data[idx][v] + (fluxes[idx][v] - fluxes[idxR][v]) * dt/dx;
      }
    }
  }
}

// Transform cell indices (i,j) into physical domain coordinates (x,y)
std::pair<double, double> X(unsigned int i, unsigned int j)
{
  const double x = i/(double)cellsX * (maxX - minX) + minX;
  const double y = j/(double)cellsY * (maxY - minY) + minY;
  return std::make_pair(x,y);
}

int main(void)
{
  clock_t start = clock();
  
  // User-defined parameters
  cellsX = 640;
  double shockSpeed = 2.95;
  double CFL = 0.9;
  
  // Domain size for shock-bubble problem.
  minX = 0;
  maxX = 1.6;
  minY = 0;
  maxY = 1.0;
  cellsY = cellsX / (maxX - minX) * (maxY - minY);
  finalT = 0.3;
  
  // Primitive variables
  State ambient = {1.0, 0, 0, 1.0};
  State bubbleInterior = {0.1, 0, 0, 1.0};
  
  // Solve Rankine-Hugoniot conditions to find state behind shock
  const double rho = ambient[RHO];
  const double p = ambient[PRE];
  const double sp2 = shockSpeed*shockSpeed;
  const double rhosp2 = rho*sp2;
  
  const double pB = (2*rhosp2-(Gamma-1.0)*p) / (1.0+Gamma);
  const double rhoB = rho*rhosp2/( rhosp2 + p - pB );
  const double uB = (pB - p)/(rho*shockSpeed) + ambient[V_X];
  
  State shockState = {rhoB, uB, 0, pB};
  
  // Set initial data:
  std::vector<State> data(cellsX * cellsY);
  std::vector<State> fluxes(cellsX * cellsY);
  
#pragma omp parallel for collapse(2)
  for(unsigned int j=0 ; j < cellsY ; j++)
  {
    for(unsigned int i=0 ; i < cellsX ; i++)
    {
      double x,y;
      std::tie(x,y) = X(i,j);
      State u;
      if(x < 0.1)
      {
        u = conservative(shockState);
      }
      else if( sqrt( pow(x-0.4, 2) + pow(y-0.5,2) ) < 0.2 )
      {
        u = conservative(bubbleInterior);
      }
      else
      {
        u = conservative(ambient);
      }
      data[i + j*cellsX] = u;
    }
  }
  
  // Do simulation
  double t = 0;
  double dx = (maxX - minX) / cellsX;
  
  while( t < finalT )
  {
    double dt = timestep(data, dx);
    dt *= CFL;
    dt = std::min(dt, finalT - t);
    for(unsigned int c=0 ; c <= 1 ; c++)
    {
      computeFluxes(data, fluxes, dx, dt, c);
      addFluxes(data, fluxes, dx, dt, c);
    }
    
    // Copy bottom row of cells into top row
    for(unsigned int i=0 ; i < cellsX ; i++)
    {
      data[i + (cellsY-1)*cellsX] = data[i+cellsX];
    }
    
    // Copy top row of cells into bottom row
    for(unsigned int i=0 ; i < cellsX ; i++)
    {
      data[i] = data[i+(cellsY-2)*cellsX];
    }
    
    t += dt;
    std::cout << "T = " << t << std::endl;
  }
  
  clock_t end = clock();
  
  std::cout << "end Time= " << (end - start) / (double)CLOCKS_PER_SEC << std::endl;
  
  // return 0;
  // Output
  
  std::ofstream output("shock_bubble.out");
  
  for(unsigned int j=0 ; j < cellsY ; j++)
  {
    for(unsigned int i=0 ; i < cellsX ; i++)
    {
      double x,y;
      std::tie(x,y) = X(i,j);
      State prim = primitive(data[j*cellsX + i]);
      output << x << " " << y << " " << prim[RHO] << " " << prim[V_X] << " " << prim[V_Y] << " " << prim[PRE] << std::endl;
    }
  }
  
  return 0;
}
