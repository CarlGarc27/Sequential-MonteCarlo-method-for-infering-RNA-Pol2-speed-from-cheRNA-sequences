# -*- coding: utf-8 -*-
"""
Created by Carlos García Ordóñez, advised by Alicia Gallego Jiménez and María Gómez Vicentefranqueira

Intelectual property of Centro de Biología Molecular (CBM) of Centro Superior de Investigaciones Científicas (CSIC) of Spain

This code implements a Sequential MonteCarlo method to infer the speed of the RNA PolII complex taking data from cheRNA sequencing.
The code also allows to generate distributions of genes sorted by speed tag (if you have that information for some genes), 
and therefore make predictions on genes you don't have a speed tag for. However, the accuracy of this is limited.
"""

#%% Import the required libraries

import csv
import os
import numpy as np
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from datetime import date

# %% Classes definitions

# Definition of an object of type Gen, which can contain all the relevant information of each gene.
class Gen:
    def __init__(self, name, chromosome, bases, counts, strand, wt_speed=None, tko_speed=None):
        """
        Function that initializes a Gen object. It requires the information about the name of the gene,
        the chromosome where it is located, the bases of its first intron, their respective reads,
        its strand, and if available, the speed tag for both Wild Type and modified cells.
        """
        
        self.chromosome = chromosome
        self.bases = bases
        self.counts = counts
        self.name = name
        self.strand = strand
        self.wt_speed = wt_speed
        self.tko_speed = tko_speed
        self.length = 0
        self.reads = 0
        self.slope = 0
        self.reads_norm_slope = 0
        self.max_norm_slope = 0
        self.lr_slope = 0
        self.observed_slope = 0
        self.lr_observed_slope = 0

    def calc_slopes(self):
        """
        Internal function of Gen objects that allows to calculate the information about the relevant
        slopes of the gene data. Should be used only in genes that already have a 'real_counts'
        attributes, which is obtained by executing the SMC method on them.
        """
        
        self.slope = (self.real_counts[-1]-self.real_counts[0])/self.length
        self.reads_norm_slope = (self.real_counts[-1]-self.real_counts[0])/(self.length*self.reads)
        self.max_norm_slope = (self.real_counts[-1]-self.real_counts[0])/(self.length*max(self.counts))
        self.lr_slope = float(linear_regression(self.binned_bases, self.real_counts)[0]*(250/self.length))
        self.observed_slope = (self.counts[-1]-self.counts[0])/self.length
        self.lr_observed_slope = float(linear_regression(self.bases, self.counts)[0])

# Definition of an object of type SMC, that can contain the important variables of the SMC process
class SMC:
    def __init__(self, particles, expected_reads, weights, real_counts, error):
        """
        Function that initializes a SMC object. It requires the information of the particles
        generated, the expected reads obtained from them, their weights, the real counts calculated
        by the SMC and the error committed by this approximation.
        """
        
        self.particles = particles
        self.expected_reads = expected_reads
        self.weights = weights
        self.real_counts = real_counts
        self.error = error

# %% Functions definitions

# Function to read the indexed text file
def read_indexed_bed_graph_file(gen_dict, gen_data_file_path, reads_data_file, delimiter="\t", delimiter2="\t"):
    """
    Function that takes 2 files as arguments, and returns a dictionary of Gen objects

    Args:
        - gen_data_file_path (string): The path to the .txt file that contains the general genes information.
        - reads_data_file (string): The parth to the .txt files that contains the total reads data per gene.
        - delimiter (char): The delimiter used in the genes data file. Default is '\t' (Tab).
        - delimiter2 (char): The delimiter used in the reads data file. Default is '\t' (Tab).

    Returns:
        - gen_dict: A dictionary where each key is a gen name and the corresponding value
              is a Gen object associated with that name.
    """

    with open(gen_data_file_path, 'r') as file:  # Open the file containing the genes data
        next(file)  # Skip the header line

        for line in file:  # For each line in the file, do:
            # Remove leading/trailing whitespace and split the line by the delimiter
            row = line.strip().split(delimiter)

            # Extract values by column
            chromosome = row[0]
            base = int(row[1])
            count = float(row[3])
            name = row[4]
            strand = row[5]
            wt_speed = row[6]
            tko_speed = row[7]

            if name not in gen_dict:
                # If the gen is not already in the dictionary, add a Gen object to it
                gen_dict[name] = Gen(name, chromosome, [base], [count], strand, wt_speed, tko_speed)
            else:
                # Otherwise, just add the bases and counts values
                gen_dict[name].bases.append(base)
                gen_dict[name].counts.append(count)

            # Set the length of the Gen
            gen_dict[name].length = len(gen_dict[name].bases)
            #if gen_dict[name].strand == '+':
            #    gen_dict[name].counts = gen_dict[name].counts[::-1]
        file.close()  # Close the file containing genes data

    with open(reads_data_file, 'r') as file2:  # Open the file containing the reads
        next(file2)  # Skip header line

        for line in file2:  # For each line in the file, do:
            # Remove leading/trailing whitespace and split the line by the delimiter2
            row = line.strip().split(delimiter2)

            # Extract the name of the gen corresponding to that line
            name = row[3]
            if name in gen_dict.keys():
                # If the name in the dictionary of genes, then add the reads value to the Gen object
                gen_dict[name].reads = float(row[5])

        file2.close()  # Close the file corresponding to the reads data

    return  # Return the completed dictionary containing Gen objects sorted by name


def write_SMC_results(gen_data, path_file, num_particles, num_tries):
    """
    Function that writes in a .csv file the predicted counts of the SMC method of a gene.

    Args:
        - gen_data (Gen object): Gen object which the data will be saved.
        - path_file (str): path of the folder where the data will be saved.
        - num_particles (int): number of particles used for the SMC method.
        - num_tries (int): number of tries used for the SMC method

    Returns:
        Nothing, as it creates the file and stores the predicted counts there.
    """

    # Open the file to be written, with the name file
    with open(path_file+'\PredictedCounts_NumPart_'+str(num_particles)+'_NumTries_'+str(num_tries)+'_'+gen_data.name, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write a single row in the file containing the predicted counts
        writer.writerow(gen_data.real_counts)
    file.close()        # Close the file. This is done for cache memory issues.
    return


def read_SMC_results(gen_data, path_file, num_particles, num_tries):
    """
    Function that reads from a .csv file the predicted counts of an SMC method of a gene of a previous experiments.
    The .csv file should be in the same format as the one produced in the function 'write_SMC_results'

    Args:
        - gen_data (Gen object): Gen object for which the predicted counts data will be recovered.
        - path_file (str): path of the folder where the data will be recovered from.
        - num_particles (int): number of particles used for the SMC method.
        - num_tries (int): number of tries for the SMC method.

    Returns
        Nothing, as the data is directly stored in the Gen object.

    """

    # Open the file from which the data will be read
    with open(path_file+'\PredictedCounts_NumPart_'+str(num_particles)+'_NumTries_'+str(num_tries)+'_'+gen_data.name, newline='') as file:
        # Set the delimiter between values to be ',', as we are reading a .csv file
        reader = csv.reader(file, delimiter=',')
        # For each row in the file (in this case should be only one), do:
        for row in reader:
            # Assing the read values to the predicted counts of the gene .
            gen_data.real_counts = row
    file.close()    # Close the file. This is done for cache memory issues.
    return


def write_SMC_process(all_SMC, name, path_file, num_particles, num_tries):
    """
    Function that writes in several .csv files the internal process of the SMC method of a gene, indexed by number of tries.
    The internal process consists of a matrix of particles, expected density reads and weights for each try of the SMC method.
    The matrix format is 'number of tries'x'gene bases'.

    Args:
        - all_SMC (list of SMC objects): list of SMC objects for which the internal SMC data will be saved.
        - name (str): name of the gene that the SMC has been applied to.
        - path_file (str): path of the folder where the data will be saved.
        - num_particles (int): number of particles used for the SMC method.
        - num_tries (int): number of tries used for the SMC method

    Returns:
        Nothing, as it creates the files and stores the internal process there.
    """

    # Iterate over the number of tries done for the SMC method
    for tries in range(num_tries):

        # Create a file for the particles matrix
        with open(path_file+'\SMC_particles'+'_NumPart_'+str(num_particles)+'_NumTries_'+str(num_tries)+'_'+name+'_Try_'+str(tries+1), 'w', newline='') as file:
            writer = csv.writer(file)
            # For each row in the matrix:
            for i in range(num_particles):
                # Write the row in the file
                writer.writerow(all_SMC[tries].particles[i])
        file.close()    # Close the file

        # Create a file for the expected density reads matrix
        with open(path_file+'\SMC_ExpDensReads'+'_NumPart_'+str(num_particles)+'_NumTries_'+str(num_tries)+'_'+name+'_Try_'+str(tries+1), 'w', newline='') as file:
            writer = csv.writer(file)
            # For each row in the matrix:
            for i in range(num_particles):
                # Write the row in the file
                writer.writerow(all_SMC[tries].expected_reads[i])
        file.close()    # Close the file

        # Create a file for the weights matrix
        with open(path_file+'\SMC_weights'+'_NumPart_'+str(num_particles)+'_NumTries_'+str(num_tries)+'_'+name+'_Try_'+str(tries+1), 'w', newline='') as file:
            writer = csv.writer(file)
            # For each row in the matrix:
            for i in range(num_particles):
                # Write the row in the file
                writer.writerow(all_SMC[tries].weights[i])
        file.close()      # Close the file
    return


def bin_data(gen_data, bin_size, data_type='Raw'):
    """
    Function that groups the counts data into bins in order to minimize outliers.

    Args:
        - gen_data (Gen object): the gene for which the bins will be made.
        - bin_size (int): the size of the bins that will be made.
        - data_type (str): the type or data you are doing the bins on. Default is Raw, altenative value is Smoothed.

    Returns:
        Nothing, as the method performs the changes directly on the Gen object.
    """

    # Compute the number of bins, the nearest integer to gene length/bin size
    num_bins = int(np.ceil(gen_data.length / bin_size))
    # Initialize an array [1,2,...,number of bins]
    gen_data.binned_bases = np.linspace(1, num_bins, num_bins)
    # Initialize an array with zeros that will contain the values for each bin
    gen_data.binned_counts = np.zeros(num_bins)
    # For each bin, do:
    for i in range(num_bins):
        # Define at what base the bin starts, which is just the number of the bin multiplied by the size of the bin
        start_index = i * bin_size
        # Define at what base the bin ends, that is either the number of the bin+1 multiplied by the size of the gen, or the last base of the gen, in case we are in the last bin
        end_index = min((i + 1) * bin_size, gen_data.length)
        # If data type is 'Raw', do:
        if data_type == 'Raw':
            # Define the value at the bin to be the mean of the observed counts between the start and the end of the bin
            gen_data.binned_counts[i] = np.mean(
                gen_data.counts[start_index:end_index])
        # If on the contrary the data is smoothed, do:
        elif data_type == 'Smoothed':
            # Define the value at the bin to be the mean of the smoothed counts between the start and the end of the bin
            gen_data.binned_counts[i] = np.mean(
                gen_data.smoothed_counts[start_index:end_index])
    return


def percentage_zeros(gen_data_counts,name):
    """
    Function that detects if a gene has either too many zeros in total or too many consecutive zeros in the observed counts, and discards it if it does.


    Args:
        - gen_data (Gen object): the gene to be analized.

    Returns:
         Nothing, as it performs the changes directly on the Gen object.

    """

    # Initialize total number of zeros to 0.
    count = 0
    # Initialize consecutive number of zeros to 0.
    consecutive_count = 0
    # Create a loop over all the binned counts
    for i in range(len(gen_data_counts)):
        # If the observed read at position i is 0, do the following:
        if gen_data_counts[i] == 0:
            # Sum 1 to the total number of zeros;
            count += 1
            # Sum 1 to the number of consecutive zeros;
            consecutive_count += 1
            # If the number of consecutive zeros is more that 5% of the data, delete the gene.
            if consecutive_count >= len(gen_data_counts)/20:
                del gen_dict[name]
                return consecutive_count
        # If the observed read at position i is not 0, set the number of consecutive zeros to 0.
        else:
            consecutive_count = 0
    # After iterating along all the gene length, if the number of zeros is higher than 10% of the gene length, delete the gene.
    if count >= len(gen_data_counts)/10:
        del gen_dict[name]
    return count


def eliminate_zeros(counts):
    """
    Function that removes zeros in the counts data by adding a small quantity to everything.

    Args:
        - counts (array): gene from which the zero counts will be eliminated.

    Returns:
        Nothing, as it performs the changes directly in the given array.
    """

    max_value = max(counts)
    for i in range(0, len(counts)):
        counts[i] += max_value/1000000

    return


def smooth_data(gen_data, smoothing_range):

    gen_data.smoothed_counts = np.zeros(gen_data.length)
    #weighteds = np.concatenate((range(1,1+int(np.ceil(smoothing_range/2))),[1+int(np.ceil(smoothing_range/2))],range(1,1+int(np.ceil(smoothing_range/2)))[::-1]))
    for i in range(gen_data.length):
        start_index = max(0, i-int(np.ceil(smoothing_range/2)))
        end_index = min(i+int(np.floor(smoothing_range/2)), gen_data.length)
        # if start_index==0:
        #    gen_data.smoothed_counts[i]=np.average(gen_data.counts[start_index:end_index+1],weights=weighteds[int(np.ceil(smoothing_range/2))-i:])
        # elif end_index==gen_data.length:
        #    gen_data.smoothed_counts[i]=np.average(gen_data.counts[start_index:end_index+1],weights=weighteds[:int(np.ceil(smoothing_range/2))-(i-gen_data.length)])
        # else:
        gen_data.smoothed_counts[i] = np.average(
            gen_data.counts[start_index:end_index+1])
    return


def compute_weight(data, mu, sigma):
    """
    Function that computes the likelihood of the data following a given normal distribution N(mu,sigma).

    Args:
        - data (float): The value that will be evaluated.
        - mu (float): The mean of the normal distribution.
        - sigma (float): The standar deviation of the normal distribution.
    Returns:
        - likelihood (float): The likelihood of the given data.
    """

    likelihood = norm.pdf(data, loc=mu, scale=sigma)
    return likelihood


def residual_systematic_resampling(weights, num_particles):
    """
    Function that performs a RSR of the sample, based on the weights given.

    A Residual Systematic Resampling is a process where you obtain a new sample 
    from an existing one, by assigning each component of the sample a weight.

    Args:
        - weights (array of floats): array of weights assigned to the sample.
        - num_particles (int): number of particles that the SMC method is using.

    Returns
        - residual_count (array of ints): array that determines how many copies of each
            particle will be resampled. Should add up to the number of particles.
    """
    
    # Initialize an array of length 'num_particles' with 0's corresponding to the residues
    u = np.zeros(num_particles)
    
    # Calculate the first value of the array from a uniform distribution on the interval [0,1/num_particles]
    u[0] = np.random.uniform(low=0, high=1/num_particles, size=1)
    
    # Initialize an array of ints of length 'num_particles' with 0's corresponding to how many times a given particle will be resampled.
    residual_count = np.zeros(num_particles, dtype=int)

    # Create a loop that iterates over the number of particles:
    for m in range(num_particles):
        # Calculate the number of times that particle will be resampled given by the RSR formula
        residual_count[m] = np.floor((weights[m]-u[m])*num_particles)+1
        # If we are not in the last step, calculate the next value of the residues given again by the RSR formula
        if m!=num_particles-1:
            u[m+1] = u[m]+(residual_count[m]/num_particles)-weights[m]
        
    # Return the array that contains how many copies of each particles will be done
    return residual_count


def linear_regression(bases, counts):
    """
    Function that calculates the linear regression parameters of the given data of a gene.

    Args:
        - bases (list of ints): list containing the bases of the gene, corresponding to the X-axis.
        - counts (list of floats): list containing the observed/predicted counts of the gene, corresponding to the Y-axis.

    Returns:
        A list of 2 elements, containing both the slope and the intercept of the linear regression
    """

    model = LinearRegression()        # Initialize the model to be a linear regression
    # Fit the linear regression to the data: bases (X-axis) vs counts (Y-axis)
    model.fit(np.array(bases).reshape(-1, 1), counts)
    # Return the slope and the intercept of the linear regression
    return [model.coef_, model.intercept_]


def abs_error(gen_data, real_counts):
    error = np.abs(gen_data.binned_counts-real_counts)
    return sum(error)


def quadratic_error(gen_data, real_counts):
    error = np.power(gen_data.binned_counts-real_counts, 2)
    return sum(error)


def positive_error(gen_data, real_counts):
    sum_error = 0
    for i in range(len(gen_data.binned_counts)):
        if real_counts[i]-gen_data.binned_counts[i] > 0:
            sum_error += real_counts[i]-gen_data.binned_counts[i]
    return sum_error


def positive_quadratic_error(gen_data, real_counts):
    sum_error = 0
    for i in range(len(gen_data.binned_counts)):
        if real_counts[i]-gen_data.binned_counts[i] > 0:
            sum_error += np.power(real_counts[i]-gen_data.binned_counts[i], 2)
    return sum_error


def data_plotting(gen_data, spline=False,  fitted_values=None):
    """
    Function that plots the data of a gene.

    Args:
        - gen_data (Gen object): Gen object to be represented.
        - spline (True/False, Optional): variable to know whether to represent the smooth spline of the observed reads or not. Default is False, to not show the spline.
        - fitted_values (list of floats, Optional): list containing the data of the smooth spline. Should be added if spline is set to True- Default value is None.

    Returns
        Nothing, as it directly shows in screen the plot with all the data
    """

    # Plots the gene binned observed counts in blue in a bar plot
    plt.bar(gen_data.binned_bases, gen_data.binned_counts[::-1], color='blue')
    if spline == True:        # If the spline is set to True, plots the spline in red in a line
        plt.plot(gen_data.binned_bases, np.exp(
            fitted_values)[::-1], color='red')
    # Plots the gene predicted counts in black in a bar plot. Alpha value is set for transparency
    plt.bar(gen_data.binned_bases,
            gen_data.real_counts[::-1], color='black', alpha=0.5)
    plt.xlabel("Bases")     # Sets the X axis name to 'Bases'
    plt.ylabel("Counts")    # Sets the Y axis name to 'Counts'
    # Sets the title to include the name of the gene, the number of particles and the number of tries used for the SMC method, the bin size, and the slope of the linear regression done on the predicted values
    plt.title("Measured reads of gene " + gen_data.name + " in blue, expected reads in black\nNumber of particles: {}".format(num_particles) +
              " Number of tries: {}".format(num_tries)+"   Bin size: {}".format(int(np.floor(gen_data.length/250)))+"\nSlope parameter: {}".format(round(gen_data.lr_slope, 15)))
    plt.show()  # Displays the plot on screen
    return


def derivative_of_SMC(gen_data):
    """
    Function that calculates the derivative of the shape of the predicted counts of a gene.

    Args
        - gen_data (Gen object): Gen object for which the derivative will be done.

    Returns
        An array of floats containing the value of the derivative at each base of the gene.
    """

    # Calculates the derivative of the predicted counts using the 'gradient' method in NumPy
    return np.gradient(gen_data.real_counts)


# %% Method to calculate the parameters needed to initialize the SMC method

def calculate_parameters(gen_data, bin_type='Proportional', bin_size=250):
    """
    FUnction that calculates the hyperparameters for the distributions that appear in the definition
    of the SMC method. For that, it first calculates a smoothing spline on the observed reads with a 
    high smoothing parameter. With that, we calculate the parameters:
        - Alpha, correspoding for the multiplicative error committed when going from one base to another.
          The smoother the reads are, the lower this becomes.
        - Residue mean and residue standar deviation, corresponding to the error committed on measuring.
        - mu_0 and sigma_0, corresponding to the value of the initial particle generation of the SMC method.

    Args:
        - gen_data (Gen object): Gen object for which the parameters and smoothing spline will be computed.
        - bin_type (str): Word that determines which type of bins will be made. Default is 'Proportional', which makes 'bin_size' number of genes, no matter how long the gene is. Could also be 'Constant', which will make bins of constant 'bin_size' length until it covers the gene; and 'None', and in that case no binning is done.
        - bin_size (int): Depending on what type of bin you choose, could mean the numbers of bins the gene will be divided into (bin type 'Proportional') or the length of the bins (bin type 'Constant'). If bin_type is chose to be None, this parameter doesn't affect the function. Defaul is 250 for a 'Proportional' bin type. 

    Returns:
        Nothing, at it stores the calculated parameters and smoothing spline values directly on the Gen object
    """
    
    # Smooth the data. Uncomment below line to activate smoothing.
    # smooth_data(gen_data,50)

    
    # Bin the data depending on the bin type chosen:
    # If bin_type is Proportional, call the bin_data function with your gene data, and a bin size calculated as the nearest integer to the gene of the length/number of bins (which is what bin_size means with this type of bins)
    if bin_type == 'Proportional':
        bin_data(gen_data, int(np.floor(gen_data.length/bin_size)), 'Raw')

    # If bin_type is Constant, call the bin_data function with your gene data and the bin size equal to bin_size
    elif bin_type == 'Constant':
        bin_data(gen_data, bin_size, 'Raw')

    # If the bin type is None, meaning we don't bin the data, do the same process that will be explained but using the gene observed counts instead of the binned counts
    elif bin_type == 'None':

        # If there is any 0 in the counts, execute the function to eliminate them
        if 0 in gen_data.counts:
            eliminate_zeros(gen_data.counts)

        # Take the logarithm of the observed reads
        log_counts = np.log(gen_data.counts)
        # Take the variance of those logarithm values
        log_var = np.std(log_counts)**2

        # Fit a cubic smoothing spline to the logarithm data, with a high smoothing parameter s (should be higher that the length of the gene in order to be appropiately smooth)
        spline_fit = UnivariateSpline(gen_data.bases, log_counts, k=3, s=len(gen_data.counts)*log_var)
        # Get the value of the calculated smoothing spline
        fitted_values = spline_fit(gen_data.bases)

        # Calculate the difference between the logarithm values and the values of the smoothing spline
        residues = log_counts - fitted_values
        # Set residue mean to be the mean of those differences
        residue_median = np.mean(residues)
        # Set residues standard deviation to be the absolute value of the standard deviation of those differences
        residue_stand_dev = np.abs(np.std(residues))

        # If f(n) denotes the value of the smoothing spline at a base n, calculate the difference e^f(n+1)-e^f(n), which is an approximation of the first particle generation
        x_n = np.exp(fitted_values[1:]) - np.exp(fitted_values[:-1])
        x_n = np.append(x_n, np.exp(fitted_values[-1]) - np.exp(spline_fit(gen_data.bases[-1] + 1)))
        # Set mu_0 to be the mean of those values
        mu_0 = np.mean(x_n)
        # Set sigma_0 to be the standard deviation of those values
        sigma_0 = np.std(x_n)

        # Calculate the difference of the logarithm of consecutive values of x_n, which serve as an approximation of the error committed when passing from one particle to the next
        log_x_n = np.log(np.abs(x_n[:-1])) - np.log(np.abs(x_n[1:]))
        # Set alpha the be the standard deviation of those values
        alpha = np.std(log_x_n)

        # Return an array containing both all the parameters and the values of the calculated smoothing spline
        return [np.array([residue_median, residue_stand_dev, alpha, mu_0, sigma_0]), fitted_values]

    # If bin_type is something different from 'Proportional', 'Constant' or 'None', raise an error informing of the appropiate values for bin_type
    else:
        print('Error, invalid bin type. Bin type should be either "Proportional", "Constant" or "None".')
        return 'Error'

    # If the bin type is either 'Proportional' or 'Constant' execute the same operations as above but with binned counts instead of observed counts:

    # If there is any 0 in the binned counts, execute the function to eliminate them
    if 0 in gen_data.binned_counts:
        eliminate_zeros(gen_data.binned_counts)

    # Take the logarithm of the binned counts
    log_counts = np.log(gen_data.binned_counts)
    # Take the variance of the logarithm values
    log_var = np.std(log_counts)**2

    # Fit a cubic smoothing spline to the logarithm data, with a high smoothing parameter s (should be higher that the number of bins in order to be appropiately smooth)
    spline_fit = UnivariateSpline(
        gen_data.binned_bases, log_counts, k=3, s=len(gen_data.binned_counts)*log_var)
    # Get the values of the calculated smoothing spline
    fitted_values = spline_fit(gen_data.binned_bases)

    # Calculate the residues of the spline
    residues = log_counts - fitted_values

    # Calculate the difference between the logarithm values and the values of the smoothing spline
    residues = log_counts - fitted_values
    # Set residue mean to be the mean of those differences
    residue_median = np.mean(residues)
    # Set residues standard deviation to be the absolute value of the standard deviation of those differences
    residue_stand_dev = np.abs(np.std(residues))

    # If f(n) denotes the value of the smoothing spline at a base n, calculate the difference e^f(n+1)-e^f(n), which is an approximation of the first particle generation
    x_n = np.exp(fitted_values[1:]) - np.exp(fitted_values[:-1])
    x_n = np.append(x_n, np.exp(
        fitted_values[-1]) - np.exp(spline_fit(gen_data.binned_bases[-1] + 1)))
    # Set mu_0 to be the mean of those values
    mu_0 = np.mean(x_n)
    # Set sigma_0 to be the standard deviation of those values
    sigma_0 = np.std(x_n)

    # Calculate the difference of the logarithm of consecutive values of x_n, which serve as an approximation of the error committed when passing from one particle to the next
    log_x_n = np.log(np.abs(x_n[:-1])) - np.log(np.abs(x_n[1:]))
    alpha = np.std(log_x_n)

    # Save the calculated parameters into the Gen object
    gen_data.hyperparameters=np.array([residue_median, residue_stand_dev, alpha, mu_0, sigma_0])
    # Save the calculated smoothing spline values
    gen_data.spline_fitted_values=fitted_values
    
    return


# %% Definition of SMC method


def SMC_method(gen_data, num_particles, num_tries=5, plotting=False):
    """
    Function that performs a Sequential MonteCarlo (SMC) method on the observed/binned counts of
    a gene in order to statistically infer the "real" counts that should be observed if we had
    maximum precision. 

    The method consists of an initial step, and then on a sequence of repetitive steps, 
    each one of them depending on the preceding one, and an evaluation of how good each step is.

    Firstly, we will work with 3 different types of data:
        - 'Particles': expressed by x_n, they represent the probability of a Pol II complex
                       being at a given base/bin n.
        - 'Expected reads': expressed by r_n, they can be interpreted as the observed reads
                            at a base/bin n that we would obtain if we had maximum precision.
        - 'Observed reads': the observed reads at a base/bin n obtained in an experiment.


    The method goes in this way. For the initial step, n=1, we assume that the first particles
    follow a normal distribution N(mu_0, sigma_0). Then a first evaluation on how good these initial
    guesses are is done. We do so by modeling the error we commit when measuring step n as following:


            log(y_n) = log(r_n) + e_n, where e_n follow a normal distribution N(residues_mean, residues_stand_dev)


    This equation means that the observed reads are just the expected reads multiplied by
    a number that is quite close to 1 + e^(residues_stand_dev).

    We then select the best guesses, and use them to create the new particles. The way one steps
    relates to the next one is as follows:


        log(x_n) = log(x_{n+1}) + v_n, where v_n follows a normal distribution N(0,alpha)


    This latter equation means that each probability is equal to the previous one, multiplied by
    a number close to 1. Therefore the error is multiplicative. Moreover, clearly the observed reads
    should then be determined by the following equation:


            expected reads at base/bin n := r_n = x_1 + x_2 + ... + x_n


    This allow us to use the first equation to check how good the guesses at step n, and again 
    select the best ones. We proceed this way until the modeling has been applied to all the bases/bins. 



    Args:
        - gen_data (Gen object): gene for which the Sequential MonteCarlo method will be applied.
        - num_particles (int): the number of particles that will be generated at each step. The higher this number the higher the precision of the method, but a value higher that 100 should suffice in general.
        - num_tries (int): the number of times the SMC method will be executed. This is done because the SMC method has a lot of 'inercy', meaning that a statistically abnormal value that is too low/high at the beginning of the method can cause high errors at the end. By executing the method multiple times these statistical anomalies are mitigated. Defaul value is 5 executions of SMC.
        - plotting (logical value): should either be True or False. Default is False. If turned to True, after the prediction of the expected reads, the method plots a bar graph showing both the observed reads and the predicted ones, and even the smoothing spline applied to the observed reads if the plotting_data function is set to do so.
    
    Returns:
        - all_SMC (list of SMC objects): list containing a SMC object for each execution of the SMC method. These objects contain the particles, expected reads, weights and predicted reads produced by the SMC method, and the error committed by it.
    """
    
    # Get the hyperparameters residue_mean/stand_dev, mu_0, sigma_0 and alpha for the gene
    parameters=gen_data.hyperparameters 
    
    # Set the length of the data to be the number of bins
    length = len(gen_data.binned_counts)
    
    # Initialize an array of zeros of that length, that will contain the final predicted counts
    gen_data.real_counts = np.zeros(length)
    
    # Initialize an empty list that will contain each of the SMC method processes
    all_SMC = []

    # Execute the SMC method 'num_tries' times
    for tries in range(num_tries):
        # Set the random generator to change over each iteration of the method
        np.random.seed(4+tries)
        
        # Initialize 3 matrices of dimension 'num_particles' x 'length' that will store the particles, expected reads and their associated weights
        particles = np.zeros((num_particles, length))
        weights = np.zeros((num_particles, length))
        expected_reads = np.zeros((num_particles, length))
        
        # Initialize 2 more matrices of dimension 'num_particles' x 'length' that will allow us to replicate the particles and expected reads that fit the data best
        particles_repl = np.zeros((num_particles, length))
        expected_reads_repl = np.zeros((num_particles, length))


        ### BEGINNING OF THE SMC METHOD ITSELF
        
        ## Start of the Initial Step:
        # Randomly choose the first particles from a normal distribution N(mu_0, sigma_0)
        particles[:, 0] = np.random.normal(parameters[3], parameters[4], num_particles)
        
        # As explained in the description of the function, the first expected reads are just the first particles
        expected_reads[:, 0] = np.abs(particles[:, 0])
        
        # Initialize the weights to be all 1/'num_particles'
        weights[:, 0].fill(1/num_particles)

        # For each particle, do:
        for j in range(num_particles):
            # Compute the weight of that particle, using the compute_weight function and that log(y_1)-log(r_1)=log(y_1/r_1) follows a normal distribution N(residue_mean, residue_stand_dev)
            weights[j, 0] = compute_weight(np.log(np.abs(gen_data.binned_counts[0]/expected_reads[j, 0])), parameters[0], parameters[1])
        
        # If all of the weights are 0, raise an error indicating that all weights were found to be 0 on step 1
        if any(weights[:, 0]) == False:
            raise ValueError('Error, weights are all zero on step 1')
            
        # Normalize the weights so they add up to 1, by dividing them by their sum
        weights[:, 0] /= np.nansum(weights[:, 0],)

        # With the weights normalized, use the RSR function to see which particles fit best and replicate them for the step 2
        new_indices = residual_systematic_resampling(weights[:,0], num_particles)
        
        # Create a multiple loop over all positions of the particles and expected reads matrices, and copy into the replacements matrices the best fits determined by the new indices.
        for q in range(1):
            pos = 0
            for m in range(num_particles):
                for k in range(new_indices[m]):
                    if pos < num_particles:
                        particles_repl[pos, q] = particles[m, q]
                        expected_reads_repl[pos,q] = expected_reads[m, q]
                        pos += 1
        
        # Set the particles and expected reads matrices to be the corresponding replacement matrices that contains the best fits
        for q in range(1):
            for pos in range(sum(new_indices)):
                particles[pos, q] = particles_repl[pos, q]
                expected_reads[pos, q] = expected_reads_repl[pos, q]

        ## Start of the recurrent step:
        # Create a loop that iterate along the lenght of the gene, doing the process described in the description of the function
        for i in range(1, length):
            # For each particle, 
            for j in range(num_particles):
                # Using that log(x_n)=log(x_{n+1}) + v_n, set the particle in the next step to be x_{n+1}=e^(log(x_n)+v_n)
                particles[j, i] = np.exp(np.log(np.abs(particles[j, i - 1])) + np.random.normal(0, parameters[2], 1))
                # Using that r_{n+1}= x_1 + x_2 + ... + x_{n+1}, calculate the expected reads at step n+1
                expected_reads[j, i] = expected_reads[j,i - 1] + particles[j, i]
                # Compute the weight of the next particle, using the compute_weight function and that log(y_n)-log(r_n)=log(y_n/r_n) follows a normal distribution N(residue_mean, residue_stand_dev)
                weights[j] = compute_weight(np.log(gen_data.binned_counts[i]/expected_reads[j, i]), parameters[0], parameters[1])
            
            # If all the computed weights are 0, raise an error informing that the problem was found at step n+1
            if any(weights[:, i]) == False:
                raise ValueError('Error, weights are all zero on step {}'.format(i+1))
                
            # Normalize the weights so they add up to 1, by dividing them by their sum
            weights[:, i] /= np.nansum(weights[:, i])

            # With the weights normalized, use the RSR function to see which particles fit best and replicate them for the next step
            new_indices = residual_systematic_resampling(weights[:,i], num_particles)

            # Create a multiple loop over all positions of the particles and expected reads matrices, and copy into the replacements matrices the best fits determined by the new indices.
            for q in range(i+1):
                pos = 0
                for m in range(num_particles):
                    for k in range(new_indices[m]):
                        if pos < num_particles:
                            particles_repl[pos, q] = particles[m, q]
                            expected_reads_repl[pos, q] = expected_reads[m, q]
                            pos += 1
            
            # Set the particles and expected reads matrices to be the corresponding replacement matrices that contains the best fits
            for q in range(i+1):
                for pos in range(sum(new_indices)):
                    particles[pos, q] = particles_repl[pos, q]
                    expected_reads[pos, q] = expected_reads_repl[pos, q]
        ### END OF THE SMC METHOD
        
        # Set the predicted counts at each base/bin to be the mean of all the expected reads at that base/bin
        real_counts = np.nanmean(expected_reads, axis=0)
        
        # Add to the 'all_SMC' list an object of type SMC, containing the information of the particles, expected reads and weights matrices, the predicted counts and the quadratic error of those predicted counts with respect to the observed ones
        all_SMC.append(SMC(particles, expected_reads, weights, real_counts, quadratic_error(gen_data, real_counts)))

    ## Once the SMC method has run 'num_tries' of times, end the function by do these final steps:
    # Initialize the sum of the inverse of the errors committed by the SMC method to be 0
    sum_errors = 0
    for tries in range(num_tries):
        gen_data.real_counts += all_SMC[tries].real_counts *(1/all_SMC[tries].error)
        sum_errors += (1/all_SMC[tries].error)
    gen_data.real_counts /= sum_errors
    
    # Call the internal Gen object calc_slopes function to calculate the different slopes information of the gene
    gen_data.calc_slopes()
    
    # Call the functions to write both the SMC process and SMC results on .csv files, so they can be recovered at a posterior time
    write_SMC_process(all_SMC, gen_data.name, SMC_process_path, num_particles, num_tries)
    write_SMC_results(gen_data, SMC_result_path, num_particles, num_tries)
    # In a file that contain the predicted slopes, write the slopes of the gene. By default only the linear regression slope is writen, but you can add more types of slopes if needed
    file = open(slopes_path+"\PredictedSlopes_NumPart_"+str(num_particles)+"_Num_Tries_"+str(num_tries), 'a')
    file.write(str(gen_data.name)+'\t'+str(gen_data.lr_slope)+'\n')
    file.close()
    
    # If the plotting setting is set to True, call the function to plot the gene data
    if plotting == True:
        data_plotting(gen_data, spline=False, gen_data.spline_fitted_values) # The default mode to call the function is without showing the smoothing spline. Switch it to true if you wish to show the smoothing spline.

    # Return the list of all the SMC objects that contain all the information of the SMC methods
    return all_SMC


# %% Create dictionary of genes from the BED files.

# Create an empty dictionary that will contain the Gen objects indexed by their name.
gen_dict = {}

# Change this to the path of your file corresponding to the Crick strand
gen_crick_file_path = r"C:\Users\CBM\CarlosJAEIntro\Datos\cheRNA\BED\WT_crick2_intron1_uniq_clean_sort_nonover_5351_5Kb_exonfilter_norm_Genes_atributes_rate.txt"
# Change this to the path of your file corresponding to the Watson strand
gen_watson_file_path = r"C:\Users\CBM\CarlosJAEIntro\Datos\cheRNA\BED\WT_watson2_intron1_uniq_clean_sort_nonover_5351_5Kb_exonfilter_norm_Genes_atributes_rate.txt"
# Change this to the path of the file containing the reads of every gene
reads_file_path = r"C:\Users\CBM\CarlosJAEIntro\Datos\cheRNA\BED\cheRNA_RefSeq_LongList_Normalized_WT.bed"

# Add the genes of the crick helix from your specified file to the gene dictionary
read_indexed_bed_graph_file(gen_dict, gen_crick_file_path, reads_file_path)
# Add the genes of the watson helix from your specified file to the gene dictionary
read_indexed_bed_graph_file(gen_dict, gen_watson_file_path, reads_file_path)

# Create a loop that iterates over all the genes, doing:
for name in gen_dict.keys():
    # If the gene strand is '+', which corresponds to Watson strand, reverse the order of its reads so they are in the right direction
    if gen_dict[name].strand=='+':
        gen_dict[name].counts=gen_dict[name].counts[::-1]
        
# %% Create dictionary with genes sorted by speed

# Create are dictionary whose keys are the different tags of speed
sorted_gen_dict = {'WT_fast': [], 'WT_medium': [], 'WT_slow': []}

# For each gene, 
for name in gen_dict:
    # If the gene speed is fast, add the gene name to the WT_fast list
    if gen_dict[name].wt_speed == 'WT_fast':
        sorted_gen_dict['WT_fast'].append(name)
    # If the gene speed is medium, add the gene name to the WT_fast list
    elif gen_dict[name].wt_speed == 'WT_medium':
        sorted_gen_dict['WT_medium'].append(name)
    # If the gene speed is slow, add the gene name to the WT_fast list
    elif gen_dict[name].wt_speed == 'WT_slow':
        sorted_gen_dict['WT_slow'].append(name)

# %% Create folder for saving today's results
today = date.today()          # Save today's date

# Create the main folder for saving the main results, with its name being today's date:
    
# Change this to the path of the main directory where you want to save your results
main_directory_path = r"C:\Users\CBM\CarlosJAEIntro\Datos\cheRNA\Resultados"
# Merge your specified main directory with today's date
main_folder_path = os.path.join(main_directory_path, str(today))
# Create that directory
os.mkdir(main_folder_path)

# Create a folder for the SMC results of the genes:
    
# Name for the folder that will contain the SMC predicted counts results
SMC_result = "Predicted_counts"
SMC_result_path = os.path.join(main_folder_path, SMC_result)
# Create a folder in the main directory with that name
os.mkdir(SMC_result_path)

# Create a folder for the SMC internal process of each one of the genes and tries:
    
# Name for the folder that will contain the SMC internal process, which include particle generation, expected density reads and their associated weights
SMC_process = "SMC_internal_process"
SMC_process_path = os.path.join(main_folder_path, SMC_process)
# Create a folder in the main directory with that name
os.mkdir(SMC_process_path)

# Create a folder for storing all the slopes calculated from linear regression on the SMC of each gene:
    
# Name for the folder that will contain the slope of the linear regression performed on the result of SMC
slopes_file = "Predicted_slopes"
slopes_path = os.path.join(main_folder_path, slopes_file)
# Create a folder in the main directory with that name
os.mkdir(slopes_path)

# %% Eliminate the genes that have too many 0's on their reads

# Create a list with all the genes names
genes_names=list(gen_dict.keys())

# Create a loop iterating over all the genes names, calling the percentage_zeros function, that checks if either the gene has too many consecutive counts that are 0 or too many total counts that are 0, and eliminates those genes.
for name in genes_names:
    percentage_zeros(gen_dict[name].counts,name)
    
# %% Calculate hyperparameters needed for the SMC method

# Create a loop that iterates over all the genes, calling the calculate_parameters function to obtain their corresponding hyperparameters.
for name in gen_dict.keys():
    calculate_parameters(gen_dict[name],bin_type='Proportional',bin_size=250)

# %% Execute the SMC method for all genes

# Initialize an empty list that will contain the name of the genes for which the SMC method has already been applied
already_done_genes = []

num_particles = 250     # Set this to the number of particles you want to run the method with
num_tries = 5           # Set this to the number of iterations of the method you want to run

# For all the genes in the dictionary, do:
for name in gen_dict:
    # If the gene has not real_counts attribute, then the SMC method has not been applied to it yet, so save its name in gens dictionary, print it and apply the SMC method to it
    if hasattr(gen_dict[name],'real_counts')==False:
        SMC_method(gen_dict[name], num_particles, num_tries, plotting=False)
        already_done_genes.append(name)
        print(name)

# %% Generate the boxplots of the genes length, sorted by speed tag, in order to visualize any possible trends
count = 0
lengths = []
len_labels = []
for speed in sorted_gen_dict.keys():
    lengths.append([])
    for name in sorted_gen_dict[speed]:
        lengths[count].append(len(gen_dict[name].counts))

    len_labels.append(speed+': {}'.format(len(lengths[count])))
    count += 1

colors = ['Red', 'purple', 'blue']
# Creating boxplot
len_bplot = plt.boxplot(lengths, notch=True,
                        patch_artist=True, labels=len_labels, showfliers=False)
for patch, color in zip(len_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Length of the genes")
plt.title("Box graph of gene length, sorted by gene speed")
plt.show()

# %% Generate the boxplots of the relevant slopes associated with the genes, sorted by speed tag, in order to visualize any possible trends
slopes = []
max_slopes = []
reads_slopes = []
lr_slopes = []
obs_slopes = []
lr_obs_slopes = []

slopes_labels = []
max_labels = []
reads_labels = []
lr_labels = []
obs_labels = []
lr_obs_labels = []

count = 0
for speed in sorted_gen_dict.keys():
    slopes.append([])
    max_slopes.append([])
    reads_slopes.append([])
    lr_slopes.append([])
    obs_slopes.append([])
    lr_obs_slopes.append([])

    for name in sorted_gen_dict[speed]:
        slopes[count].append(gen_dict[name].slope)
        max_slopes[count].append(gen_dict[name].max_norm_slope)
        reads_slopes[count].append(gen_dict[name].reads_norm_slope)
        lr_slopes[count].append(gen_dict[name].lr_slope)
        obs_slopes[count].append(gen_dict[name].observed_slope)
        lr_obs_slopes[count].append(gen_dict[name].lr_observed_slope)

    slopes_labels.append(speed+': {}'.format(len(slopes[count])))
    max_labels.append(speed+': {}'.format(len(max_slopes[count])))
    reads_labels.append(speed+': {}'.format(len(reads_slopes[count])))
    lr_labels.append(speed+': {}'.format(len(lr_slopes[count])))
    obs_labels.append(speed+': {}'.format(len(obs_slopes[count])))
    lr_obs_labels.append(speed+': {}'.format(len(lr_obs_slopes[count])))
    count += 1


colors = ['Red', 'purple', 'blue']

# Creating plot
slopes_bplot = plt.boxplot(
    slopes, notch=True, patch_artist=True, labels=slopes_labels, showfliers=False)
for patch, color in zip(slopes_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title("Box graph of slopes calculated of SMC method, sorted by gene speed")
plt.show()
# Creating plot
max_bplot = plt.boxplot(max_slopes, notch=True,
                        patch_artist=True, labels=max_labels, showfliers=False)
for patch, color in zip(max_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title(
    "Box graph of Max Normalized slopes calculated by SMC method, sorted by gene speed")
plt.show()
# Creating plot
reads_bplot = plt.boxplot(reads_slopes, notch=True,
                          patch_artist=True, labels=reads_labels, showfliers=False)
for patch, color in zip(reads_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title("Box graph of Reads Normalized slopes calculated by SMC method, sorted by gene speed")
plt.show()
# Creating plot
obs_bplot = plt.boxplot(obs_slopes, notch=True,
                        patch_artist=True, labels=obs_labels, showfliers=False)
for patch, color in zip(obs_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title("Box graph of slopes calculated of observed reads, sorted by gene speed")
plt.show()

# Creating plot
lr_obs_bplot = plt.boxplot(lr_obs_slopes, notch=True,
                           patch_artist=True, labels=lr_obs_labels, showfliers=False)
for patch, color in zip(lr_obs_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title("Box graph of  LR slopes performed on observed reads, sorted by gene speed")
plt.show()

# Creating plot
lr_bplot = plt.boxplot(lr_slopes, notch=True,
                       patch_artist=True, labels=lr_labels, showfliers=False)
for patch, color in zip(lr_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title("Box graph of LR slopes performed on SMC method, sorted by gene speed")
plt.show()

# %% Generate the boxplots of the different error values committed by the SMC method on the genes, sorted by speed tag, in order to visualize any possible trends
SMC_error = []
SSRes_error = []
SST_error = []
RS_error = []
AdjRS_error = []

error_labels = []
SSRes_labels = []
SST_labels = []
RS_labels = []
AdjRS_labels = []

count = 0
for speed in sorted_gen_dict.keys():
    SMC_error.append([])
    SSRes_error.append([])
    SST_error.append([])
    RS_error.append([])
    AdjRS_error.append([])
    for name in sorted_gen_dict[speed]:
        if gen_dict[name].reads != 0:
            gen_dict[name].SMC_error = quadratic_error(
                gen_dict[name], gen_dict[name].real_counts)
            SMC_error[count].append(gen_dict[name].SMC_error)

            X = np.array(gen_dict[name].binned_bases).reshape(-1, 1)
            y = gen_dict[name].real_counts
            model = LinearRegression()
            model.fit(X, y)
            yhat = model.predict(X)
            SS_Residual = sum((y-yhat)**2)
            SS_Total = sum((y-np.mean(y))**2)
            if SS_Total != 0 and gen_dict[name].reads != 0:
                r_squared = 1 - (float(SS_Residual))/SS_Total
                adjusted_r_squared = 1 - \
                    (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)

                SSRes_error[count].append(SS_Residual/gen_dict[name].reads)
                SST_error[count].append(SS_Total/gen_dict[name].reads)
                RS_error[count].append(r_squared)
                AdjRS_error[count].append(adjusted_r_squared)

    error_labels.append(speed+': {}'.format(len(SMC_error[count])))
    SSRes_labels.append(speed+': {}'.format(len(SSRes_error[count])))
    SST_labels.append(speed+': {}'.format(len(SST_error[count])))
    RS_labels.append(speed+': {}'.format(len(RS_error[count])))
    AdjRS_labels.append(speed+': {}'.format(len(AdjRS_error[count])))
    count += 1


colors = ['Red', 'purple', 'blue']

error_bplot = plt.boxplot(
    SMC_error, notch=True, patch_artist=True, labels=error_labels, showfliers=False)
for patch, color in zip(error_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title("Box graph of mean error comitted by SMC, sorted by gene speed")
plt.show()

SSRes_error_bplot = plt.boxplot(
    SSRes_error, notch=True, patch_artist=True, labels=SSRes_labels, showfliers=False)
for patch, color in zip(SSRes_error_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title(
    "Box graph of SS Residuals comitted by Linear Regression, sorted by gene speed")
plt.show()

SST_error_bplot = plt.boxplot(
    SST_error, notch=True, patch_artist=True, labels=SST_labels, showfliers=False)
for patch, color in zip(SST_error_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title("Box graph of SST comitted by Linear Regression, sorted by gene speed")
plt.show()

RS_error_bplot = plt.boxplot(
    RS_error, notch=True, patch_artist=True, labels=RS_labels, showfliers=False)
for patch, color in zip(RS_error_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title(
    "Box graph of R-squared comitted by Linear Regression, sorted by gene speed")
plt.show()

AdjRS_error_bplot = plt.boxplot(
    AdjRS_error, notch=True, patch_artist=True, labels=AdjRS_labels, showfliers=False)
for patch, color in zip(AdjRS_error_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Slopes of the genes")
plt.title("Box graph of adjusted R-squared comitted by Linear Regression, sorted by gene speed")
plt.show()

# %% Histogram plot in log scale of linear regression slopes and errors, sorted by gene speed, to check if they follow a normal distribution
num_intervals = 65

figure, axis = plt.subplots(1, 2)
figure.set_size_inches(20, 10)

axis[0].hist(np.log(lr_slopes[2]), bins=num_intervals,
             density=True, color='blue')
axis[0].plot(np.histogram(np.log(lr_slopes[2]), bins=num_intervals, density=True)[1], norm.pdf(np.histogram(np.log(
    lr_slopes[2]), bins=num_intervals, density=True)[1], np.mean(np.log(lr_slopes[2])), np.std(np.log(lr_slopes[2]))), color='red')

axis[0].hist(np.log(lr_slopes[0]), bins=num_intervals,
             density=True, color='black', alpha=0.75)
axis[0].plot(np.histogram(np.log(lr_slopes[2]), bins=num_intervals, density=True)[1], norm.pdf(np.histogram(np.log(
    lr_slopes[2]), bins=num_intervals, density=True)[1], np.mean(np.log(lr_slopes[0])), np.std(np.log(lr_slopes[0]))), color='purple')

axis[1].hist(np.log(SMC_error[2]), bins=num_intervals,
             density=True, color='blue')
axis[1].plot(np.histogram(np.log(SMC_error[2]), bins=num_intervals, density=True)[1], norm.pdf(np.histogram(np.log(
    SMC_error[2]), bins=num_intervals, density=True)[1], np.mean(np.log(SMC_error[2])), np.std(np.log(SMC_error[2]))), color='red')

axis[1].hist(np.log(SMC_error[0]), bins=num_intervals,
             density=True, color='black', alpha=0.75)
axis[1].plot(np.histogram(np.log(SMC_error[0]), bins=num_intervals, density=True)[1], norm.pdf(np.histogram(np.log(
    SMC_error[0]), bins=num_intervals, density=True)[1], np.mean(np.log(SMC_error[0])), np.std(np.log(SMC_error[0]))), color='purple')

plt.show()

# %% Boxplot of the second derivatives of the expected reads, sorted by speed tag, in order to see if there are any trends. Also contains a visual verification that convexity follows a normal distribution
count = 0
convexity = []
convexity_labels = []
for speed in sorted_gen_dict.keys():
    convexity.append([])
    for name in sorted_gen_dict[speed]:
        if name != 'Fam168b' and name != '1700019D03Rik' and name != 'Anxa7' and name != '4931414P19Rik':
            # if np.mean(np.gradient(np.gradient(gen_dict[name].real_counts,edge_order=2),edge_order=2))<0.5*(10**-10):
            convexity[count].append(np.log(np.abs(np.mean(np.gradient(
                np.gradient(gen_dict[name].real_counts, edge_order=2), edge_order=2)))))

    convexity_labels.append(speed+': {}'.format(len(convexity[count])))
    count += 1

figure, axis = plt.subplots(1, 2)
figure.set_size_inches(24, 10)

axis[0].hist(convexity[0], 100, density=True, color='red')
axis[0].hist(convexity[1], 100, density=True, color='purple')
axis[0].hist(convexity[2], 100, density=True, color='blue')


axis[1].hist(convexity[0], bins=100, density=True, color='blue')
axis[1].plot(np.histogram(convexity[0], bins=100, density=True)[1], norm.pdf(np.histogram(
    convexity[0], bins=100, density=True)[1], np.mean(convexity[0]), np.std(convexity[0])), color='red')
#axis[0].hist(np.log(lr_slopes[1]), bins=num_intervals,density=True, color='red')
axis[1].hist(convexity[2], bins=100, density=True, color='black', alpha=0.75)
axis[1].plot(np.histogram(convexity[2], bins=100, density=True)[1], norm.pdf(np.histogram(
    convexity[2], bins=100, density=True)[1], np.mean(convexity[2]), np.std(convexity[2])), color='purple')

plt.show()

# Creating plot
convexity_bplot = plt.boxplot(
    convexity, notch=True, patch_artist=True, labels=convexity_labels, showfliers=False)
for patch, color in zip(convexity_bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Genes speeds groups")
plt.ylabel("Convexity of the SMC method of the genes")
plt.title(
    "Box graph of mean of second derivative of SMC method, sorted by gene speed")
plt.show()

convx_distributions = []
for i in range(3):
    convx_distributions.append(
        norm(np.mean(convexity[i]), np.std(convexity[i])))
    print(convx_distributions[i].pdf(np.mean(convexity[0])), convx_distributions[i].pdf(
        np.mean(convexity[1])), convx_distributions[i].pdf(np.mean(convexity[2])))
    
# %% Prediction of the speed tags for new genes. It is necessary that the speed tag for some genes is already known
percentage = 10
prueba = 1
results = np.zeros((9, 100))
while percentage < 100 and prueba <= 100:

    genes_training_names = []
    lr_training = []
    lr_labels_training = []
    SMC_training = []
    convexity_training = []
    slopes_training = []
    SST_error_training = []
    count = 0

    for speed in sorted_gen_dict.keys():
        possible_names = list(sorted_gen_dict[speed])
        lr_training.append([])
        slopes_training.append([])
        SMC_training.append([])
        convexity_training.append([])
        SST_error_training.append([])
        h = 0
        while h < int(len(gen_dict)*(percentage/300)):
            np.random.seed(prueba)
            name = np.random.choice(possible_names)
            possible_names.remove(name)
            if gen_dict[name].reads != 0:
                genes_training_names.append(name)
                lr_training[count].append(gen_dict[name].lr_slope)
                slopes_training[count].append(gen_dict[name].lr_observed_slope)
                SMC_training[count].append(gen_dict[name].SMC_error)
                convexity_training[count].append(np.log(np.abs(np.mean(np.gradient(
                    np.gradient(gen_dict[name].real_counts, edge_order=2), edge_order=2)))))
                SST_error_training[count].append(
                    sum((gen_dict[name].real_counts-np.mean(gen_dict[name].real_counts))**2))
                h += 1

        lr_labels_training.append(speed+': {}'.format(len(lr_training[count])))
        count += 1

    slopes_distributions = []
    error_distributions = []
    convexity_distributions = []
    SST_error_distributions = []
    for i in range(3):
        slopes_distributions.append(
            norm(np.mean(np.log(lr_training[i])), np.std(np.log(lr_training[i]))))
        error_distributions.append(
            norm(np.mean(np.log(SMC_training[i])), np.std(np.log(SMC_training[i]))))
        convexity_distributions.append(
            norm(np.mean(convexity_training[i]), np.std(convexity_training[i])))
        SST_error_distributions.append(norm(
            np.mean(np.log(SST_error_training[i])), np.std(np.log(SST_error_training[i]))))

    number_tries = 1
    accuracy_total = []
    accuracy_fast = []
    accuracy_slow = []
    accuracy_medium = []

    for k in range(number_tries):
        np.random.seed(prueba*k)
        observed_frequency = {'WT_fast': 0, 'WT_medium': 0, 'WT_slow': 0}
        expected_frequency = {'WT_fast': 0, 'WT_medium': 0, 'WT_slow': 0}
        genes_test = {}
        accuracy_extremes_fast = 0
        accuracy_extremes_slow = 0
        quartile_fast = 0
        quartile_slow = 0
        i = 0
        for name in gen_dict.keys():
            if name not in genes_training_names and gen_dict[name].reads != 0:
                probabilities = []

                if gen_dict[name].wt_speed == 'WT_fast' and gen_dict[name].lr_slope >= np.quantile(lr_training[0], [0, 0.33, 0.66, 1])[2]:
                    quartile_fast += 1
                elif gen_dict[name].wt_speed == 'WT_slow' and gen_dict[name].lr_slope <= np.quantile(lr_training[2], [0, 0.33, 0.66, 1])[1]:
                    quartile_slow += 1

                for j in range(3):
                    probabilities.append(slopes_distributions[j].pdf(np.log(gen_dict[name].lr_slope))*error_distributions[j].pdf(np.log(gen_dict[name].SMC_error))*convexity_distributions[j].pdf(np.log(np.abs(np.mean(
                        np.gradient(np.gradient(gen_dict[name].real_counts, edge_order=2), edge_order=2)))))*SST_error_distributions[j].pdf(np.log(sum((gen_dict[name].real_counts-np.mean(gen_dict[name].real_counts))**2))))

                probabilities = np.power(probabilities, 1)
                probabilities /= sum(probabilities)

                # if probabilities[0]>0.45 or probabilities[1]>0.45 or probabilities[2]>0.45:
                if True == True:
                    # probabilities=np.power(probabilities,2)/np.sum(np.power(probabilities,2))
                    """
                    result=np.random.multinomial(1, probabilities, size=1)
                    if result[0][0]==1:
                        genes_test[name]='WT_fast'
                    elif result[0][1]==1:
                        genes_test[name]='WT_medium'
                    elif result[0][2]==1:
                        genes_test[name]='WT_slow'
                    """
                    if max(probabilities) == probabilities[0]:
                        genes_test[name] = 'WT_fast'
                    elif max(probabilities) == probabilities[1]:
                        genes_test[name] = 'WT_medium'
                    elif max(probabilities) == probabilities[2]:
                        genes_test[name] = 'WT_slow'

                i += 1

        accuracy_total.append(0)
        accuracy_fast.append(0)
        accuracy_slow.append(0)
        accuracy_medium.append(0)
        for name in genes_test.keys():
            observed_frequency[genes_test[name]] += 1
            expected_frequency[gen_dict[name].wt_speed] += 1
            if gen_dict[name].wt_speed == genes_test[name]:
                accuracy_total[k] += 1
                if genes_test[name] == 'WT_fast':
                    accuracy_fast[k] += 1
                    if gen_dict[name].lr_slope >= np.quantile(lr_training[0], [0, 0.33, 0.66, 1])[2]:
                        accuracy_extremes_fast += 1
                elif genes_test[name] == 'WT_slow':
                    accuracy_slow[k] += 1
                    if gen_dict[name].lr_slope <= np.quantile(lr_training[2], [0, 0.33, 0.66, 1])[1]:
                        accuracy_extremes_slow += 1
                elif genes_test[name] == 'WT_medium':
                    accuracy_medium[k] += 1

    # print(len(genes_test))
    # accuracy_total[k]/=len(genes_test)
    # accuracy_fast[k]/=expected_frequency['WT_fast']
    # accuracy_slow[k]/=expected_frequency['WT_slow']
    results[int(percentage/10)-1][prueba -
                                  1] = round(np.mean(accuracy_total)/len(genes_test)*100, 2)
    #print('Percentage training test',str(round(len(genes_training_names)/len(gen_dict)*100))+'%', 'Global accuracy '+str(round(np.mean(accuracy_total)/len(genes_test)*100,2))+'%')
    #print('Accuracy on extremes, quartile 33%', 'Fast_'+str(round(accuracy_extremes_fast/quartile_fast*100,2))+'%', 'Slow_'+str(round(accuracy_extremes_slow/quartile_slow*100,2))+'%')
   # print('Global accuracy by speed','Fast_'+str(round(np.mean(accuracy_fast)/observed_frequency['WT_fast']*100,2))+'%', 'Slow_'+str(round(np.mean(accuracy_slow)/observed_frequency['WT_slow']*100,2))+'%','\n')
    # print('Efectividad:',np.mean(accuracy_total)/len(genes_test),np.mean(accuracy_fast)/observed_frequency['WT_fast'],np.mean(accuracy_slow)/observed_frequency['WT_slow'])
    #print('Especificidad:' ,np.mean(accuracy_total)/len(genes_test),np.mean(accuracy_fast)/expected_frequency['WT_fast'],np.mean(accuracy_slow)/expected_frequency['WT_slow'])
    percentage += 10
    if percentage > 90:
        percentage = 10
        prueba += 1

for i in range(9):
    plt.hist(results[i], bins=20)
    plt.show()

plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90], np.mean(results, axis=1))
# %% Representation of the accuracy of the prediction. Only possible to calculate if the real speed tag for the genes predicted is available
figure, axis = plt.subplots(1, 2)
figure.set_size_inches(14, 7)

axis[0].hist(accuracy_total, bins=20, density=True, color='blue')
axis[0].plot(np.histogram(accuracy_total, bins=20, density=True)[1], norm.pdf(np.histogram(
    accuracy_total, bins=20, density=True)[1], np.mean(accuracy_total), np.std(accuracy_total)), color='red')

axis[1].hist(accuracy_fast, bins=20, density=True, color='blue')
axis[1].plot(np.histogram(accuracy_fast, bins=20, density=True)[1], norm.pdf(np.histogram(
    accuracy_fast, bins=20, density=True)[1], np.mean(accuracy_fast), np.std(accuracy_fast)), color='red')

axis[1].hist(accuracy_slow, bins=20, density=True, color='green')
axis[1].plot(np.histogram(accuracy_slow, bins=20, density=True)[1], norm.pdf(np.histogram(
    accuracy_slow, bins=20, density=True)[1], np.mean(accuracy_slow), np.std(accuracy_slow)), color='black')

plt.show()

# %% Generation of images representing the expected reads, in order to use Machine Learning to predict new one. Speed tag for the used genes must be known

etiquettes_file = r"C:\Users\CBM\CarlosJAEIntro\Datos\cheRNA\Resultados\Etiquettes_file"
imagenes_path = os.path.join(main_folder_path, 'Imagenes')
os.mkdir(imagenes_path)
file = open(imagenes_path+"\Etiquettes_file", "w")
file.write('Name\t Speed Group\t Strand\n')
for name in gen_dict.keys():
    file.write(name+'\t'+gen_dict[name].wt_speed +
               '\t'+str(gen_dict[name].strand)+'\n')
    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(5, 5)
    axis.plot(gen_dict[name].binned_bases,
              gen_dict[name].real_counts[::-1], color='black')
    axis.axis('off')
    plt.savefig(imagenes_path+"\Gen_"+name, dpi=100)
    plt.close()
file.close()

# %% Escribir temporalmente los resultados
