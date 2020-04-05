/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include <chrono>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;
using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles


	// initialize all particles' positions and weights
  for(int i=0; i<num_particles; i++)
	{
		Particle particle;

		particle.id = i+1;

		// sample data from Gaussian distribution (considering Gaussian noise)
		addGaussianNoise(particle,x,y,theta,std);
		
		particle.weight = 1.0;
		weights.push_back(particle.weight);

		particles.push_back(particle);

	}

	// set is_initialized to true
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	// epsilon to avoid dividing by zero
	double eps = 1e-16;

	// compute travelled distance between two time steps
	double delta_s = velocity/(yaw_rate+eps);

	// compute change in heading
	double delta_theta = yaw_rate*delta_t;

	for(int i=0; i<num_particles; i++)
	{	
		particles[i].x = particles[i].x + delta_s*(sin(particles[i].theta+delta_theta)-sin(particles[i].theta)); 
		particles[i].y = particles[i].y + delta_s*(cos(particles[i].theta)-cos(particles[i].theta+delta_theta));
		particles[i].theta = fmod((particles[i].theta + delta_theta), 2.0*M_PI);

		// add Gaussian noise on the resulting position/heading estimates
		addGaussianNoise(particles[i],particles[i].x,particles[i].y,particles[i].theta,std_pos);

	}
}

void ParticleFilter::dataAssociation(Particle& particle,
																		 Map& map_landmarks, 
                                     LandmarkObs& observation) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
	vector<double> d_list;

	// calculate distance between certain observation in map coordinates and landmark positions within sensor range
	for(int i=0; i<map_landmarks.landmark_list.size(); i++)
	{
		double d = dist(observation.x, observation.y, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f); 	
		d_list.push_back(d);

	}

	// assign nearest landmark to observation
	vector<double>::iterator result = std::min_element(d_list.begin(), d_list.end());

	int nearest_neighbor_index = std::distance(d_list.begin(), result);

	observation.id = map_landmarks.landmark_list[nearest_neighbor_index].id_i;

	// assign association and associated landmark coordinates (world frame) to particle
	particle.associations.push_back(observation.id);
	particle.sense_x.push_back(map_landmarks.landmark_list[nearest_neighbor_index].x_f);
	particle.sense_y.push_back(map_landmarks.landmark_list[nearest_neighbor_index].y_f);


	// delete landmark assigned from list of landmarks
	map_landmarks.landmark_list.erase(map_landmarks.landmark_list.begin()+nearest_neighbor_index);

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

	// clear value of weights of particle filter
	weights.clear();

	// sum of all particles weights
	double sum = 0.;

	// calculate predicted landmark measurements for each particle
	for(int i=0; i<num_particles; i++)
	{

		double weight = 1.0;

		// clear particles associations and associated landmark coordinates	
		particles[i].associations.clear();
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();


		// initialize homogenous transformation matrix
		MatrixXd A(3,3);
		A << cos(particles[i].theta), -sin(particles[i].theta), particles[i].x,
				 sin(particles[i].theta), cos(particles[i].theta), particles[i].y,
				 0, 0, 1;

		// inverse of homogenous transformation matrix
		MatrixXd Ai = A.inverse();

		// copy of actual map landmarks
		// operate only on this data in data association
		Map current_map_landmarks = map_landmarks;

		// variable counting number of landmarks beyond sensor range
		int num_erased = 0;

		// consider only landmarks within sensor range
		// transformation of landmark positions from map frame to car frame necessary
		for(int l=0; l<map_landmarks.landmark_list.size(); l++)
		{
			VectorXd landmark_map(3);

			// assign coordinates of current landmark (w.r.t. map frame) to vector
			landmark_map(0) = map_landmarks.landmark_list[l].x_f;
			landmark_map(1) = map_landmarks.landmark_list[l].y_f;
			landmark_map(2) = 1;

			// apply an inverse homogenous transformation to map landmark coordinates from map frame to car frame
			VectorXd landmark_car = Ai*landmark_map;

			if(dist(landmark_car(0),landmark_car(1),0.,0.)>sensor_range)
			{
				current_map_landmarks.landmark_list.erase(current_map_landmarks.landmark_list.begin()+l-num_erased);
				num_erased++;	
			}

		}


		for(int j=0; j<observations.size(); j++)
		{
			if(current_map_landmarks.landmark_list.size()>0)
			{ 
				VectorXd meas_car(3);

				// assign coordinates of current observation (w.r.t. car frame) to vector
				meas_car(0) = observations[j].x;
				meas_car(1) = observations[j].y;
				meas_car(2) = 1;

				// apply a homogenous transformation to map observation coordinates from vehicle frame to map frame
				VectorXd meas_map = A*meas_car;

				// store transformed observation coordinates (w.r.t. map frame) to LandmarkObs object
				LandmarkObs obs_m;
				obs_m.x = meas_map(0);
				obs_m.y = meas_map(1); 


				// call nearest neighbor data association
				dataAssociation(particles[i], current_map_landmarks, obs_m);

				// computation of weight
				weight *= multiv_prob(std_landmark[0], std_landmark[1], obs_m.x, obs_m.y, map_landmarks.landmark_list[obs_m.id-1].x_f, map_landmarks.landmark_list[obs_m.id-1].y_f); 

			}
			
		}
		
		// assign computed weight to corresponding particle
		particles[i].weight = weight;

		sum += particles[i].weight;		

		// Set assocations for visualization purpose only 
		SetAssociations(particles[i], particles[i].associations, particles[i].sense_x, particles[i].sense_y);

	}

	// normalizing each particles weight
	for(int i=0; i<num_particles; i++)
	{
		particles[i].weight /= sum;
		weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

	std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(weights.begin(),weights.end());
  int p[num_particles] = {};
  for(int n=0; n<num_particles; n++) {
  	++p[dist(gen)];
  }
	
	vector<Particle> new_particles;

	for(int n=0; n<num_particles; n++)
	{
			std::fill_n (std::back_inserter(new_particles), p[n], particles[n]);
	}

	particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

void ParticleFilter::addGaussianNoise(Particle& particle,
																			double x, double y,
																			double theta, double std[]) {
	default_random_engine gen;

	// set a seed depending on system clock
	gen.seed(std::chrono::system_clock::now().time_since_epoch().count());

	// normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);
  
  // normal distributions for y
  normal_distribution<double> dist_y(y, std[1]);

	// normal distribution for theta
  normal_distribution<double> dist_theta(theta, std[2]);

	//sample from normal distribution where "gen" is the random engine initialized earlier
	particle.x = dist_x(gen);
	particle.y = dist_y(gen);
	particle.theta = dist_theta(gen);

}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}


