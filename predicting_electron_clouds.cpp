#include "Eigen/Dense"
#include <vector>
#include <random>
#include <fstream>
#include "NeuralNetwork.h"

int main()
{
	std::vector<uint32_t> layer_sizes = {7500,3000,1000,500,1000,3000,7500};
	NeuralNetwork autoencoder(layer_sizes);
	
	//The input file consists of the images of the graphene structures with various defects
	//The output file consists of the probability density n(x,y) for the free electons in matrix form
	//The resolution in each case is (100,75)
	//This gives an input vector of size 7500
	std::ifstream inputs_file("defective_graphene_inputs.txt");
	std::ifstream outputs_file("defective_graphene_outputs.txt");
	
	std::vector<DataPoint> training_data;
	training_data.reserve(1000);
	
	for(uint32_t i=0;i<1000;i++){
		Eigen::VectorXf inputs(7500);
		Eigen::VectorXf expected_outputs(7500);	
		for(uint32_t j=0;j<7500;j++){
			inputs_file >> inputs(j);
			outputs_file >> expected_outputs(j);
		}
		training_data.emplace_back(inputs,expected_outputs);
	}
	
	autoencoder.train(training_data,2000,10,0.01);
	
	// Test the autoencoder on some seen and unseen samples
	std::ifstream testing_inputs("defective_graphene_testing_inputs.txt");
	std::ofstream testing_outputs("defective_graphene_testing_outputs.txt");
	
	for(uint32_t i=0;i<1000;i++){
		Eigen::VectorXf inputs(7500);
		for(uint32_t j=0;j<7500;j++){
			testing_inputs >> inputs(j);
			testing_outputs << autoencoder.calculate_outputs(inputs) << '\n';
		}
	}
}