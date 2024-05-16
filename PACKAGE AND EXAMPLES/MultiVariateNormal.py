import numpy as np
import pandas as pd
from scipy.stats import f, chi2
import matplotlib.pyplot as plt

# If only dipersion matrix is given
class MVN_Dispersion_Matrix :

    # Creating instance of the class 'MVN_Dispersion_Matrix'
    def __init__(self, var_cov_matrix):
        self.var_cov_matrix = np.array(var_cov_matrix)
        std_dev = np.sqrt(np.diag(self.var_cov_matrix))
        corr_matrix = np.divide(self.var_cov_matrix,np.outer(std_dev,std_dev))
        self.correlation_matrix = np.round(corr_matrix,2)

    # name 
    def __name__(self) :
        return "MVN(var_cov_matrix)"

    # str
    def __str__(self) :
        return "It is an object of Multivariate Nomral distibution instantiated with only variance-covariance matrix"     
    # repr
    def __repr__(self) :
        return "It is an object of Multivariate Nomral distibution instantiated with only variance-covariance matrix"
    
    # Independency between any two one dimensional vaiables
    def linear_independent(self,Coeff1 = [],Coeff2 = []) :
        '''Displays the given two one-dimensional variables are whether independent or not\n\n
        Coeff1 - list of values of coefficients yields variable 1 --> list()\n
        Coeff2 - list of values of coefficients yields variable 2 --> list()\n\n
        Example - If X = [X1,X2,X3], then independency of X1 and X2 + X3 can be verified by giving Coeff1 = [1,0,0] and Coeff2 = [0,1,1]'''
        if Coeff1 == [[]] or Coeff2 == [[]] :
            raise ValueError("Coeff1 and Coeff2 cannot be empty")
        T = np.array([Coeff1])
        M = np.array([Coeff2]).transpose()
        cov = np.dot(np.dot(T,self.var_cov_matrix),M)
        if cov == 0 :
            print("Independent")
        else :
            print("Not independent")
    
    # Independency between any two P-dimensional component of vaiables
    def multi_independent(self,component1 = [],component2 = []) :
        '''Displays the given two P-dimensional variables are whether independent (or) not\n\n
        component1 - list of values of indices of first component --> list()\n
        component2 - list of values of indices of second component--> list()\n\n
        Example - If X = [X1,X2,X3], then independency of [X1,X2] and [X3] can be verified by giving component1 = [1,2] and component2 = [3]'''
        if component1 == [] or component2 == [] :
            raise ValueError("component1 and component2 cannot be empty")
        component1 = [x-1 for x in component1 ]
        component2 = [x-1 for x in component2 ]
        cov = self.var_cov_matrix[component1][:,component2] 
        if np.all(cov == 0) :
            print("Independent")
        else :
            print("Not independent") 

    def partial_correlation(self,i = [],constant = []) :
        '''Returns the correlation between two variables while keeping set of variable(s) as constant\n\n
        i - list of indices of variables to find the correlation between --> list()\n
        constant - the indice of the variable(s) to be constant --> list()\n\n
        Example - if X = [X1,X2,X3,X4], the partial correlation of X1 and X2 keeping X3 and X4 constant can be obtained by giving i = [1,2] and constant = [3,4]'''
        if i == [] or constant == [] :
            raise ValueError("i and constant cannot be empty")
        given = [ 0 for x in constant]
        mean = [0 for x in range(self.var_cov_matrix.shape[0])]
        mvn = MultiVariateNorm(mean, self.var_cov_matrix)
        var_cov_matrix = mvn.conditional(i,constant,given).var_cov_matrix
        return np.round(var_cov_matrix[0][1]/np.sqrt(var_cov_matrix[0][0]*var_cov_matrix[1][1]),2)

    # Multiple correlation coefficient
    def multiple_correlation(self,x = None,independent_set = []):
        '''Returns the correlation between one dependent variable and independebt set of other variables\n\n
        x - indices of variable to find the correlation with the remaing set of variables --> int()\n
        independent_set - list of indices independent variables, thus we want to find the correlation of x and this set of independent variabls --> list()\n\n 
        Example - if X = [X1,X2,X3,X4,X5], the multiple correlation of X1 and [X2,X3,X4,X5]  can be obtained by giving x = 1'''
        if x == None or independent_set == [] :
            raise ValueError("x and independent_set cannot be empty")
        x = x - 1
        var_x = self.var_cov_matrix[x][x]
        set = [ y-1  for y in independent_set ]
        var_set = self.var_cov_matrix[set][:,set] 
        cov = self.var_cov_matrix[x][set]
        mutiple_corr_num = np.dot(np.dot(cov,np.linalg.inv(var_set)),cov.transpose())
        return np.round(np.sqrt(mutiple_corr_num / var_x),2)

# MultiVariateNormal distribution with mean and variance covariance matrix 
    
class MultiVariateNorm(MVN_Dispersion_Matrix) :
    # Creating instance of the class 'MultiVariateNorm'
    def __init__(self,mean_vector,var_cov_matrix) :
        if type(mean_vector) == np.ndarray :
            self.mean_vector =  mean_vector
            self.var_cov_matrix = var_cov_matrix
        else :
            self.mean_vector = np.array([[i] for i in mean_vector])
            self.var_cov_matrix = np.array(var_cov_matrix)  
        super().__init__(self.var_cov_matrix)

    # name 
    def __name__(self) :
        return "MVN(mean_vector,var_cov_matrix)"
    
    # str
    def __str__(self) :
        return "It is an object of Multivariate Nomral distibution instantiated with mean vector and variance-covariance matrix"
    # repr
    def __repr__(self) :
        return "It is an object of Multivariate Nomral distibution instantiated with mean vector and variance-covariance matrix"
        
    # Getting Basic description about the object of <class 'MultiVariateNormal'> 
    def info(self) : 
        '''Displays P-variate Mutivariate normal distribution with mean and variance'''
        print(f"{len(self.mean_vector)}-Variate Normal Distribution".center(100," "),end = "\n\n")
        print(f"Mean vector".center(100,"-"),f"\n{np.round(self.mean_vector,2)}")
        print(f"Dispersion matrix".center(100,"-"),f"\n{np.round(self.var_cov_matrix,2)}")
    
    # Marginal distribution
    def marginal(self,i = []) :
        '''Returns Marginal Distribution of variable/components of variables given in i\n\n
        i - list of variable(s) indices--> list()\n\n
        Example - if X = [X1,X2,X3,X4] then marginal distribution of X1 can be obtained by giving i = [1] and [X2,X3] can be obtained by giving i = [2,3]'''
        if i == [] :
            raise ValueError("i cannot be empty")
        i = [x-1 for x in i]
        mar_mean = self.mean_vector[i] 
        mar_variance = self.var_cov_matrix[i][:,i]
        return MultiVariateNorm(mar_mean,mar_variance)
    
    # Conditional distribution
    def conditional(self,i = [], j = [],given_values_j = []) :
        '''Returns conditional distrbution of variable in i given variables in j and Displays the estimates of the parameters of the conditional distribution if the given_values_j is not given\n\n
        i - list of variable indices of first component --> list()\n
        j - list of variable indices of second component --> list()\n
        given_values_j - list of values of given variable(s) --> list()\n\n
        Example - If X_1 = [X1,X2] and X_2 = [X3,X4], then conditional distribution of X_1 / X_2 = [0,0] can be obtained by putting i = [1,2], j = [3,4]  and given_values_j = [0,0]'''
        if i == [] or j ==[] :
            raise ValueError("i and j cannot be empty")
        i = [x-1 for x in i]
        x1_mean = self.mean_vector[i] 
        x1_var = self.var_cov_matrix[i][:,i]
        j = [ y-1 for y in j]
        x2_mean = self.mean_vector[j] 
        x2_var = self.var_cov_matrix[j][:,j]
        x12_var = self.var_cov_matrix[i][:,j]
        conditional_var = x1_var - np.dot(np.dot(x12_var,np.linalg.inv(x2_var)),x12_var.transpose())
        if given_values_j == [] :
            print("Warning: Values of second component in not given\n")
            est_mean_par = np.dot(x12_var,np.linalg.inv(x2_var))
            print(f"Estimates of the Parameters of the conditional Distribution".center(100," "),end = "\n\n")
            print(f"Estimate Associated with Mean vector ".center(100,"-"),f"\n{np.round(est_mean_par,2)}")
            print(f"Estimate of Dispersion matrix".center(100,"-"),f"\n{np.round(conditional_var,2)}")
        else :      
            given = np.array([given_values_j]).transpose()
            conditional_mean = x1_mean + np.dot(np.dot(x12_var,np.linalg.inv(x2_var)),(given - x2_mean))
            return MultiVariateNorm(conditional_mean,conditional_var)
        
    
    # Covariance between any two variables (or) any two set of variables
    def covariance(self,i = [], j= []) :
        '''Returns the covariance matrix of two components\n\n
        i takes the indices of first component in a list --> list()\n
        j takes the indices of second component in a list --> list()\n\n
        Example : If X_1 = [X1,X2] and X_2 = [X3,X4], then covariance(X_1,X_2) can be obtained by giving i = [1,2] j = [3,4]'''
        if i == [] or j == []:
            raise ValueError("i and j cannot be empty")
        i = [x-1 for x in i]
        j = [y-1 for y in j]
        x12_var = self.var_cov_matrix[i][:,j]
        return x12_var
    
    # Covariance between any two set of variables multiplied by some non-singluar coefficient matrix
    def special_covariance(self,i = [],j=[],Coeff1 = [[]],Coeff2 = [[]]) :
        '''Returns the covariance matrix of two components multipled with non-singular matrices\n\n
        i takes the indices of first component in a list --> list()\n
        j takes the indices of second component in a list --> list()\n
        Coeff1 - Coefficient matrix of first component --> list(list())\n
        Coeff2 - Coefficient matrix of second component --> list(list())\n
        Example : If X_1 = [X1,X2] and X_2 = [X3,X4], then covariance(A*X_1, B*X_2) can be obtained by giving i = [1,2], j = [3,4], Coeff1 = A and Coeff2 = B'''
        if i == [] or j== [] or Coeff1 == [[]] or Coeff2 == [[]] :
            raise ValueError("i, j, Coeff1 and Coeff2 cannot be empty")
        Coeff1 = np.array(Coeff1)
        Coeff2 = np.array(Coeff2)
        i = [x-1 for x in i]
        j = [y-1 for y in j]
        x12_var = self.var_cov_matrix[i][:,j]
        return np.dot(np.dot(Coeff1,x12_var),Coeff2.transpose())
    
   
    # Findng the distribution of non-singular transformation
    def distribution(self,Coeff_matrix = [[]]):
        '''Returns the distribution of given variable (or) expression of variables (or) component of expression of variables\n\n
        Coeff_matrix - Coefficients yields the variable (or) component of expression of variables --> list(list())\n\n
        Example - If X= [X1,X2,X3] , then distribution of [[X1+X2],[X1-X2],[X1-X3]] can be obtained by giving Coeff_matrix = [[1,1,0],[1,-1,0],[1,0,-1]]'''
        if Coeff_matrix == [[]] :
            raise ValueError("Coeff_matrix cannot be empty")
        C = np.array(Coeff_matrix)
        mean_vector = np.dot(C,self.mean_vector)
        var_cov_matrix = np.dot(np.dot(C,self.var_cov_matrix),C.transpose())
        return MultiVariateNorm(mean_vector,var_cov_matrix)
     

# For reading data from .csv file and to make an instance of 'MultiVariateNorm' with mean vector and dispersion matrix
def read_data(path_or_df) :
    '''Return the Object of <class 'MultiVariateNorm'> with mean vector and dispersion matrix obtained from the data given\n\n
    path_or_df - Give the path of your file (or) pandas.DataFrame --> only (.csv)'''
    if type(path_or_df) == str :
        df = pd.read_csv(path_or_df)
    else :
        df = path_or_df    
    mean , cov =  df.mean().to_numpy().reshape(df.shape[1],1) , df.cov().to_numpy()
    return MultiVariateNorm(np.round(mean,4),np.round(cov,4))

# If only Correlation matrix is given
class MVN_Correlation_Matrix :

    # Creating instance of the class 'MVN_Correlation_Matrix'
    def __init__(self,correlation_matrix) :
        self.correlation_matrix = np.array(correlation_matrix)

    # name 
    def __name__(self) :
        return "MVN(correlation_matrix)"

    # str
    def __str__(self) :
        return "It is an object of Multivariate Nomral distibution instantiated with only correlation matrix"     
    
    # str
    def __repr__(self) :
        return "It is an object of Multivariate Nomral distibution instantiated with only correlation matrix"     
    
    # Partial correlation
    def partial_correlation(self,i = [], constant = []) :
        '''Returns the correlation between two variables while keeping set of variable(s) as constant\n\n
        i - list of indices of variables to find the correlation between --> list()\n
        constant - the indice of the variable(s) to be constant --> list()\n\n
        Example - if X = [X1,X2,X3,X4], the partial correlation of X1 and X2 keeping X3 and X4 constant can be obtained by giving i = [1,2] and constant = [3,4]'''
        if i == [] or constant == [] :
            raise ValueError("i and constant cannot be empty")
        corr = self.correlation_matrix
        i = [x-1 for x in i]
        j = [y-1 for y in constant]
        partial_num = corr[i[0]][i[1]] - sum([corr[i[0]][z]*corr[i[1]][z] for z in j])
        denom_1 = np.sqrt(1-sum([np.square(corr[i[0]][[z]]) for z in j]))[0]
        denom_2 = np.sqrt(1-sum([np.square(corr[i[1]][[z]]) for z in j]))[0]
        return np.round(partial_num/(denom_1*denom_2),2)
    
    # Multiple correlation
    def multiple_correlation(self, x = None , independent_set = []) :
        '''Returns the correlation between one dependent variable and independebt set of other variables\n\n
        x - indices of variable to find the correlation with the remaing set of variables --> int()\n
        independent_set - list of indices independent variables, thus we want to find the correlation of x and this set of independent variabls --> list()\n\n 
        Example - if X = [X1,X2,X3,X4,X5], the multiple correlation of X1 and [X2,X3,X4,X5]  can be obtained by giving x = 1'''
        if x == None or independent_set == [] :
            raise ValueError("x and independent_set cannot be empty")
        corr = self.correlation_matrix
        num_set = [x-1]
        set = [ y-1  for y in independent_set ]
        num_set.extend(set)
        num = np.linalg.det(corr[num_set][:,num_set])
        denom = np.linalg.det(corr[set][:,set])
        value = 1 - (num/denom)
        return np.round(np.sqrt(value),2)
    

# Hotelling's Test for Multivariate Normal Mean
class Hotelling_T2:
    '''Hotelling T-Square Test'''

    def one_sample(data = [[]],mean = [],n = None ,size = 0.05 , plot = False) :
        ''' One-Sample Hotelling's T-Square Test\n
        data --> path of your .csv file for large sample (or) matrix format - [[]] for small sample (or) pandas.core.frame.DataFrame\n
        mean --> testing mean enclosed in list() --> []\n
        size --> significance level --> default 0.05\n
        plot --> CAR plot(if visualable) --> Critical and Acceptance Region plot --> take boolean --> default False'''
        # Reading data in four formats
        if type(data) != MultiVariateNorm and type(data) != list and type(data) != str and type(data) != pd.core.frame.DataFrame :
            raise TypeError("Warning: data is not of any expected datatype")
        # As MultivariateNorm() object
        if type(data) == MultiVariateNorm :
            mvn = data
        else :
            # In matrix format
            if type(data) == list :
                df = pd.DataFrame(np.array(data).transpose())
            # In .csv file format
            elif type(data) == str :
                df = pd.read_csv(data)
            # In Pandas.DataFrame    
            elif type(data) == pd.core.frame.DataFrame :   
                df = data
            # sample size
            n = df.shape[0]
            # Creating a multivaraite normal object
            mvn = read_data(df)

        # gathering sample mean_vector and sample variance covariance matrix from the multivariate nomral object
        x_bar , S = mvn.mean_vector, mvn.var_cov_matrix 
        
        # no.of variables 
        p = len(mean)    
        # Given mean we want to test   
        mean = np.array([mean]).transpose()   
        # Calculating T2 statistic 
        matrix_cal = np.dot(np.dot((x_bar - mean).transpose(),np.linalg.inv(S)),x_bar - mean)
        T2 = (n*matrix_cal)[0][0]
        # F-distribution       
        f_cal = ((T2*(n-p)) / ((n-1)*p)) # calculated value
        f_table = f.ppf(1 - size,p,n-p) # table value
        # p value
        p_value = np.round(1 - f.cdf(f_cal, p, n-p),4)

        # printing results
        print("".center(100,"="))
        print("Hotelling T-Square Test".center(100," "))
        print("".center(100,"-"))
        print(f"The test statistic value is T2 : {np.round(T2,2)}")
        print(f"Level of significance : {size}")
        print(f"F distribution : F({p}, {n-p}) degrees of freedom")
        # suitable p value if p = 0.0
        if p_value == 0.0 :
            p_value = 0.0001
        print(f"p - value : {p_value}")
        if p_value <= size :
            print(f"Decision : we reject the null hypothesis with {(1-size)*100} % confidence level")
        else :
            print(f"Decision : we failed to reject the null hypothesis with {(1-size)*100} % confidence level")  
        print("".center(100,"="))    
              
        if plot :
            # Visualization
            if f_cal > 10 :
                print("Oops! Visualization is not appropriate for this problem")
            else :
                # pdf of F-distn
                x = np.linspace(0,max(f_cal + 1 ,f_table+ 1),1000)
                pdf_values = f.pdf(x,p,n-p)
                plt.plot(x,pdf_values, label = f"F({p}, {n-p})")
                # critical region
                x_critical =  np.linspace(f_table,max(f_cal + 1,f_table + 1),100)
                plt.fill_between(x_critical, f.pdf(x_critical, p, n-p), color='red', alpha = 0.3, label=f'Critical Region ({size} level)')
                # acceptance region
                x_acceptance = np.linspace(0, f_table, 100)
                plt.fill_between(x_acceptance, f.pdf(x_acceptance, p, n-p), color='green', alpha=0.3, label='Acceptance Region')
                # calculated value
                plt.axvline(f_cal, color='black', linestyle = "dashed", label=f'Calculated F-value ({f_cal:.2f})')

                # plot
                plt.xlabel('F-value')
                plt.ylabel('Probability Density Function (PDF)')
                plt.title('F-Distribution Curve with Critical and Acceptance Regions')
                plt.legend()
                plt.show()


    def two_sample(data1, data2, n1 = None , n2 = None, equal_cov = True, size = 0.05 , plot = False) :
        '''Two-sample Hotelling's T-Square Test\n
        data1 --> sample 1 --> pandas.core.frame.DataFrame(recommended)\n
        data2 --> sample 2 --> pandas.core.frame.DataFrame(recommended)\n
        n1, n2 --> give only if you have given data1 and data2 as MultiVariateNorm object --> int\n
        equal_cov --> boolean True or False if you know or you can test it itself by giving "BoxM-test"\n
        size --> significance level --> default 0.05\n
        plot --> CAR plot(if visualable) --> Critical and Acceptance Region plot --> take boolean --> default False'''
        
        # Reading data in three formats
        # As two DataFrames
        if type(data1) == pd.core.frame.DataFrame and type(data2) == pd.core.frame.DataFrame :
            population = (data1, data2)
            n1, n2 = data1.shape[0], data2.shape[0]
            data1 , data2 = read_data(data1), read_data(data2)
            
        # As two MultiVariateNorm objects 
        if type(data1) == MultiVariateNorm and type(data2) == MultiVariateNorm :
            population = (data1,data2)
            x_bar_1 , S1 = data1.mean_vector, data1.var_cov_matrix
            x_bar_2 , S2 = data2.mean_vector, data2.var_cov_matrix
            sample_size = {"n1" : n1 , "n2" : n2}
            disp_matrix = {"pop1" : S1, "pop2" : S2}

        # no of variables
        p = S1.shape[0]

        # Test for common covariance matrix 
        # Box's M test
        if equal_cov == "BoxM-test" : # doubt 
            test_size = size
            g = len(population)
            S_denom = sum([ (sample_size[f"n{i + 1}"] - 1) for i in range(g) ])
            S_numer = sum([ (sample_size[f"n{i + 1}"] - 1)*disp_matrix[f"pop{i + 1}"]  for i in range(g)]) 
            S = S_numer / S_denom
            M = S_denom*np.log(np.linalg.det(S)) -  sum([ (sample_size[f"n{i + 1}"] - 1)*np.log(np.linalg.det(disp_matrix[f"pop{i + 1}"]))  for i in range(g)])  
            u1 = sum([(1 / (sample_size[f"n{i + 1}"] - 1)) for i in range(g) ]) - (1 / S_denom)
            u2 = (2*(p**2) + 3*p - 1) / (6*(p + 1)*(g - 1))
            u = u1*u2
            M_cal = (1 - u)*M
            df = p*(p + 1)*(g - 1) / 2
            M_table = chi2.ppf(test_size,df)
            equal_cov = (M_cal <= M_table)
                
            
        # equal covariance is true    
        if equal_cov :     
            
            S = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
            # Calculating T2 statistic 
            matrix_cal = np.dot(np.dot((x_bar_1 - x_bar_2).transpose(),np.linalg.inv(S)),x_bar_1 - x_bar_2)
            T2 = ((n1*n2)*matrix_cal/(n1+n2))[0][0]
            # F-distribution       
            f_cal = (T2*(n1+n2-p-1)) / (p*(n1+n2-2)) # calculated value
            f_table = f.ppf(1- size,p,n1+n2-p-1) # table value
            
            # p value
            p_value = np.round(1 - f.cdf(f_cal, p, n1+n2-p-1),4)

            # printing results
            print("".center(100,"="))
            print("Hotelling T-Square Test".center(100," "))
            print("".center(100,"-"))
            print(f"The test statistic value is T2 : {np.round(T2,2)}")
            print(f"Level of significance : {size}")
            print(f"F distribution : F({p}, {n1+n2-p-1}) degrees of freedom")
            # suitable p value if p = 0.0
            if p_value == 0.0 :
                p_value = 0.0001 
            print(f"p - value : {p_value}")
            if p_value <= size :
                print(f"Decision : we reject the null hypothesis with {(1-size)*100} % confidence level")
            else :
                print(f"Decision : we failed to reject the null hypothesis with {(1-size)*100} % confidence level")
            print("".center(100,"="))

            if plot :
                # Visualization
                if f_cal > 10 :
                    print("Oops! Visualization is not appropriate for this problem")
                else :
                    # pdf of F-distn
                    x = np.linspace(0,max(f_cal + 1,f_table + 1),1000)
                    pdf_values = f.pdf(x, p, n1+n2-p-1)
                    plt.plot(x, pdf_values, label = f"F({p}, {n1+n2-p-1})")
                    # critical region
                    x_critical =  np.linspace(f_table,max(f_cal + 1,f_table + 1),100)
                    plt.fill_between(x_critical, f.pdf(x_critical, p, n1+n2-p-1), color='red', alpha = 0.3, label=f'Critical Region ({size} level)')
                    # acceptance region
                    x_acceptance = np.linspace(0, f_table, 100)
                    plt.fill_between(x_acceptance, f.pdf(x_acceptance, p, n1+n2-p-1), color='green', alpha=0.3, label='Acceptance Region')
                    # calculated value
                    plt.axvline(f_cal, color='black', linestyle = "dashed", label=f'Calculated F-value ({f_cal:.2f})')

                    # plot
                    plt.xlabel('F-value')
                    plt.ylabel('Probability Density Function (PDF)')
                    plt.title('F-Distribution Curve with Critical and Acceptance Regions')
                    plt.legend()
                    plt.show()

        # equal covariance is False            
        else : # Fisher Behren's method  
            if n1 == n2 : # for equal sample size
                mvn = MultiVariateNorm(x_bar_1 - x_bar_2, S1 + S2)
                test_mean = [0 for x in range(x_bar_1.shape[0])]
                Hotelling_T2.one_sample(mvn, mean = test_mean, n = n1, size = size, plot = plot)

            else :# for unequal sample size
                if n1 < n2 : 
                    mvn = MultiVariateNorm(x_bar_1 - x_bar_2, S1 + (n1/n2)*S2)
                    test_mean = [0 for x in range(x_bar_1.shape[0])]
                    Hotelling_T2.one_sample(mvn, mean = test_mean, n = n1, size = size, plot = plot) # one sample
                else :
                    mvn = MultiVariateNorm(x_bar_2 - x_bar_1, S2 + (n2/n1)*S1)
                    test_mean = [0 for x in range(x_bar_2.shape[0])]
                    Hotelling_T2.one_sample(mvn, mean = test_mean, n = n2, size = size, plot = plot) # one sample

    def pop_seperator(file_or_data, variables = None, by = None, pop = None) :
        '''A Support function for two sample Hotelling's T-Square test which is used to split data as our wish - return two Dataframes\n
        file_or_data --> .csv file or pandas.core.frame.DataFrame\n
        variables --> mention variables of interest in the data\n
        by --> the variable responsible splitting the data into two populations(recommened two categories in by = " ")\n 
        pop --> if by  = " " has more than two, using pop we can extract two populations of interest'''
        if type(file_or_data) == str :
            df = pd.read_csv(file_or_data)
        else :
            df = file_or_data    
        if pop == None : 
            temp_col = [x for x in df[by]]
            pop_col = []
            for item in temp_col:
                if item not in pop_col:
                    pop_col.append(item) 
        else :
            if type(pop) != list : 
                pop_col = list(pop)
            else :
                pop_col = pop     
        if len(pop_col) != 2 :
             raise LookupError(f"Expected two groups in 'by', but {len(pop_col)} were given")    
        # generator 
        if variables == None :
            for label in pop_col :
                yield df[df[by] == label].drop(by,axis = 1).reset_index(drop = True)
        else :
            for label in pop_col :
                yield df[df[by] == label].drop(by,axis = 1).reset_index(drop = True)[variables]
                

# Multivariate Analysis of Variance(MANOVA) 
class MANOVA :
    '''Multivariate Analysis of Variance(MANOVA) for comparing means of several Multivariate normal populations'''      
    def test(*population, manova_table = True, size = 0.05 ,plot = False) :
        '''MANOVA - Wilk's Lambda based test\n\n
        population(s) --> give the population as pandas.core.frame.DataFrame\n 
        manova_table --> boolean --> default True\n
        size --> significance level --> default 0.05\n
        --> CAR plot(if visualable) --> Critical and Acceptance Region plot --> take boolean --> default False'''
        # sample size dictionary
        sample_size = dict()
        # sample mean dictionary
        sample_mean = dict()
        # sum of squares dictionary
        ss = dict()
        # sum of cross products dictionary
        cp = dict()
        # overall mean
        overall_mean = pd.concat(population,axis = 0).mean().to_numpy()
        
        # collecting sample size and sample mean as dictionaries
        for i,df in enumerate(population) :
            sample_size[f"pop{i+1}"] = df.shape[0]
            sample_mean[f"pop{i+1}"] = df.mean().to_numpy()
            df.columns = [ x for x in range(df.shape[1])]
        
        # collecting sum of squares as dictionary    
        for i in range(population[0].shape[1]) :
            ssmean, sstrt, ssres = (0, 0, 0)
            for j,df in enumerate(population) :
                ssmean = ssmean + df.shape[0]*np.square(overall_mean[i])
                sstrt = sstrt + df.shape[0]*np.square(sample_mean[f"pop{j+1}"][i] - overall_mean[i])
                ssres = ssres + sum([ np.square(x - sample_mean[f"pop{j+1}"][i]) for x in df[i]])    
            ss[f"var{i+1}"] = {"sst" : np.round(ssmean + sstrt + ssres,2), "crt_sst" : np.round(sstrt + ssres,2), "ssmean" : np.round(ssmean,2),"sstrt" : np.round(sstrt,2),"ssres" : np.round(ssres,2)}
        
        # collecting sum of cross products as dictionary    
        for i in range(population[0].shape[1]) :
            for z in range(population[0].shape[1]) :
                if i != z and i < z :
                    cpmean , cptrt , cpres = (0,0,0) 
                    for j,df in enumerate(population) :
                        cpmean  = cpmean + df.shape[0]*(overall_mean[i]*overall_mean[z])
                        cptrt = cptrt + df.shape[0]*(sample_mean[f"pop{j+1}"][i] - overall_mean[i])*(sample_mean[f"pop{j+1}"][z] - overall_mean[z])
                        cpres = cpres + sum([(x - sample_mean[f"pop{j+1}"][i])*(df[z][k] - sample_mean[f"pop{j+1}"][z]) for k,x in enumerate(df[i])])
                    cp[f"var{i+1}{z+1}"] = { "cpt" : np.round(cpmean + cptrt + cpres,2) , "crt_cpt" : np.round(cptrt + cpres,2) ,"cpmean" : np.round(cpmean,2),  "cptrt" : np.round(cptrt,2) , "cpres" : np.round(cpres,2)} 

        # sum of squares and cross products matrices            
        treatment = np.zeros((df.shape[1],df.shape[1]))
        residual  = np.zeros((df.shape[1],df.shape[1]))
        total_crct = np.zeros((df.shape[1],df.shape[1]))
        for i in range(treatment.shape[0]) :
            for j in range(treatment.shape[1]) :
                if i == j :
                    treatment[i][j] = ss[f"var{i+1}"]["sstrt"]
                    residual[i][j] = ss[f"var{i+1}"]["ssres"]
                    total_crct[i][j] = ss[f"var{i+1}"]["crt_sst"]
                else :
                    try :
                        treatment[i][j] = cp[f"var{i+1}{j+1}"]["cptrt"]
                        residual[i][j] = cp[f"var{i+1}{j+1}"]["cpres"]
                        total_crct[i][j] = cp[f"var{i+1}{j+1}"]["crt_cpt"]
                    except KeyError :
                        treatment[i][j] = cp[f"var{j+1}{i+1}"]["cptrt"] 
                        residual[i][j] = cp[f"var{j+1}{i+1}"]["cpres"]
                        total_crct[i][j] = cp[f"var{j+1}{i+1}"]["crt_cpt"]
                        
        # printing MANOVA table            
        if manova_table :                
            print("".center(100,"="))
            print("MANOVA : Sum of Squares and Cross Product Matrix with associated Degrees of Freedom".center(100," "))
            print("".center(100,"="))
            print("Treatment".center(100,"-"))
            print(treatment, end = "    ")
            print(f"with {len(population) - 1} df", end = "\n\n")
            print("Residual".center(100,"-"))
            print(residual, end = "    ")
            print(f"with {sum(sample_size.values()) - len(population)} df",end = "\n\n")
            print("Total(corrected)".center(100,"-"))
            print(total_crct, end = "    ")
            print(f"with {sum(sample_size.values()) - 1} df",end = "\n\n")
            print("".center(100,"="),end ="\n\n")

        # wilk's lambda 
        wilks_lambda = np.linalg.det(residual) / np.linalg.det(total_crct)

        # Null distributon of wilk's lambda
        p = population[0].shape[1]
        g = len(population)
        N = sum(sample_size.values())
        a =  N - g - ((p - g + 2) / 2)
        if p**2 + (g-1)**2 - 5 > 0 :
            b = np.sqrt((p**2 * (g-1)**2 - 4)/(p**2 + (g-1)**2 - 5))
        else :
            b = 1
        c = (p*(g-1) - 2) / 2
        f_cal = ((a*b - c) /(p*(g-1))) * ((1 - np.power(wilks_lambda,(1/b))) / np.power(wilks_lambda,(1/b)))
        dof1, dof2 = p*(g-1), a*b - c
        f_tab = f.ppf(1 - size, dof1, dof2)
        p_value = 1 - f.cdf(f_cal,dof1, dof2)

        # rounding
        wilks_lambda = np.round(wilks_lambda,2)
        p_value = np.round(p_value, 4)
        
        # Printing results with Wilk's lambda
        print("".center(100,"="))
        print("Hypothesis Test Results".center(100," "))
        print("".center(100,"-"))
        print(f"The test statistic value is (wilk's lambda) : {wilks_lambda}")
        print(f"Level of significance : {size}")
        print(f"F distribution : F({int(dof1)}, {int(dof2)}) degrees of freedom")
        
        # suitable p value if p = 0.0
        if p_value == 0.0 :
            p_value = 0.0001
        print(f"p - value : {p_value}")
        if p_value <= size :
            print(f"Decision : we reject the null hypothesis with {(1 - size)*100} % confidence level")
        else :
            print(f"Decision : we failed to reject the null hypothesis with {(1 - size)*100} % confidence level")
        print("".center(100,"="))     
           
            
        if plot :
            # Visualization
            if f_cal > 10 :
                print("Oops! Visualization is not appropriate for this problem")
            else :
                # pdf of F-distn
                x = np.linspace(0,max(f_cal + 1,f_tab + 1),1000)
                pdf_values = f.pdf(x, dof1,dof2)
                plt.plot(x, pdf_values, label = f"F({int(dof1)}, {int(dof2)})")
                # critical region
                x_critical =  np.linspace(f_tab,max(f_cal + 1,f_tab + 1),100)
                plt.fill_between(x_critical, f.pdf(x_critical, dof1, dof2), color='red', alpha = 0.3, label=f'Critical Region ({size} level)')
                # acceptance region
                x_acceptance = np.linspace(0, f_tab, 100)
                plt.fill_between(x_acceptance, f.pdf(x_acceptance, dof1, dof2), color='green', alpha=0.3, label='Acceptance Region')
                # calculated value
                plt.axvline(f_cal, color='black', linestyle = "dashed", label=f'Calculated F-value ({f_cal:.2f})')

                # plot
                plt.xlabel('F-value')
                plt.ylabel('Probability Density Function (PDF)')
                plt.title('F-Distribution Curve with Critical and Acceptance Regions')
                plt.legend()
                plt.show()
                     
    def pop_seperator(file, variables = None, pop = None, by = None) :
        '''A Support function for one-way MANOVA which is used to split data as our wish - return Dataframes\n
        file_or_data --> .csv file or pandas.core.frame.DataFrame\n
        variables --> mention variables of interest in the data\n
        by --> the variable responsible splitting the data into populations\n 
        pop --> if by  = " " has more population categories, using pop we can extract suitable number populations of interest'''
        if type(file) == str :
            df = pd.read_csv(file)
        else :
            df = file        
        if pop == None : 
            temp_col = [x for x in df[by]]
            pop_col = []
            for item in temp_col:
                if item not in pop_col:
                    pop_col.append(item)
        else :
            if type(pop) != list : 
                pop_col = list(pop)
            else :
                pop_col = pop 
        if len(pop_col) <= 2 :
             raise LookupError(f"Expected more than two groups in 'by', but {len(pop_col)} were given")           
        if variables == None :
            for label in pop_col :
                yield df[df[by] == label].drop(by,axis = 1).reset_index(drop = True)
        else :
            for label in pop_col :
                yield df[df[by] == label].drop(by,axis = 1).reset_index(drop = True)[variables]       