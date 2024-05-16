# Importing all at once
from MultiVariateNormal import MultiVariateNorm, MVN_Correlation_Matrix, MVN_Dispersion_Matrix
from MultiVariateNormal import read_data, MANOVA as M
from MultiVariateNormal import Hotelling_T2 as HT2
## 1
# MANOVA is used for dividing the population 
df1, df2, df3 = M.pop_seperator("DATASETS\Iris.csv", by = "Species")
mvn1 = read_data(df1)
print("Mean vector is\n",mvn1.mean_vector,end= "\n\n")
print("Dispersion matrix is\n",mvn1.var_cov_matrix,end= "\n\n")
print("Correlation matrix is\n",mvn1.correlation_matrix,end= "\n\n")

## 2
corr_matrix = [[1.0000, 0.4248, 0.0420, 0.0215, 0.0573],[0.4248, 1.0000, 0.1487, 0.2489, 0.2843],[0.0420, 0.1487, 1.0000, 0.6693, 0.4662],[0.0215, 0.2489, 0.6693, 1.0000, 0.6915],[0.0573, 0.2843, 0.4662, 0.6915, 1.000]]
mvn2 = MVN_Correlation_Matrix(corr_matrix)
corr45_3 = mvn2.partial_correlation( i = [4,5], constant = [3])
print("The partial correlation between X4 and X5, holding X3 fixed is",corr45_3)
corr12_345 = mvn2.partial_correlation(i = [1,2], constant = [3,4,5])
print("The partial correlation between X1 and X2, holding X3, X4 and X5 fixed is",corr12_345)
corr1_345 = mvn2.multiple_correlation( x = 1 , independent_set = [3,4,5])
print("The multiple correlation between X1 and the set X3, X4 and X5 is",corr1_345)

## 3
var_cov_matrix = [[4,1,2],[1,9,-3],[2,-3,25]]
mvn3 = MVN_Dispersion_Matrix(var_cov_matrix)
corr1_23 = mvn3.multiple_correlation(x = 1, independent_set= [2,3])
print("R1.23 is",corr1_23)


## 4
mean = [130.24, 3.55, 150.44, 327.79]
var_cov_matrix = [[377.20,28.03,418.76,1008.96],[28.03,13.66,35.98,140.56],[418.76,35.98,467.91,1148.56],[1008.96,140.56,1148.56,3097.49]]
mvn4 = MultiVariateNorm(mean, var_cov_matrix)
mvn4.conditional(i =[1,4], j= [2,3], given_values_j = [4.3, 112.6]).info()
corr34_2 = mvn4.partial_correlation(i = [3,4], constant = [2])
print("The partial correlation coefficient (X3, X4) and (X2) is",corr34_2)


## 5
mean = [0,0,0]
var_cov_matrix = [[7/27,-4/27,-5/27],[-4/27,10/27,-1/27],[-5/27,-1/27,19/27]]
mvn5= MultiVariateNorm(mean, var_cov_matrix)
mvn5.info()
print("\nMarginal of X(1) is")
mvn5.marginal(i = [1,2]).info()
print("\nMarginal of X(2) is ")
mvn5.marginal(i = [3]).info()
print("\nConditional of X(1) | X(2) is")
mvn5.conditional(i = [1,2], j = [3])
print("\nConditional of X(2) | X(1) is")
mvn5.conditional(i = [3], j = [1,2])

## 6
mean = [-3, 1, 4]
var = [[1,-2,0],[-2,5,0],[0,0,2]]
mvn6 = MultiVariateNorm(mean, var)
# X1 and X2
mvn6.multi_independent(component1 = [1], component2 = [2])
# [X1, X2] and [X3]
mvn6.multi_independent(component1 = [1,2], component2 = [3])
# X2 and X2 - (5/2)X1 - X3
mvn6.linear_independent(Coeff1 = [0,1,0], Coeff2 = [-5/2,1,-1])
# Resultant distributions
mvn6.distribution(Coeff_matrix = [[1,1,0],[1,-1,0]]).info()
mvn6.distribution(Coeff_matrix = [[1,1,0],[1,-1,0],[-1,0,1]]).info()

## 7
mean = [2,4,1,3,0]
var = [[4,-1,1/2,-1/2,0],[-1,3,1,-1,0],[1/2,1,6,1,-1],[-1/2,-1,1,4,0],[0,0,-1,0,2]]
mvn7 = MultiVariateNorm(mean,var)
mvn7.info()
print("\nCov(X(1),X(2))\n", mvn7.covariance([1,2],[3,4,5]))
print("\nCov(AX(1),BX(2))\n", mvn7.special_covariance(i =[1,2],j = [3,4,5],Coeff1 = [[1,-1],[1,1]],Coeff2 =[[1,1,1],[1,1,-2]]))

## 8
HT2.one_sample("DATASETS\Female Perspiration Data.csv", mean = [4, 50, 10], size = 0.10, plot = True)

## 9
df1, df2 = HT2.pop_seperator("DATASETS\Riding Mower Owners Data.csv", by = "ownership")
## 9.1 Assuming Equal Covariance
HT2.two_sample(df1, df2, size = 0.01, equal_cov = True,  plot = True)
## 9.2 Testing Equal Covariance
HT2.two_sample(df1, df2, size = 0.01, equal_cov = "BoxM-test",  plot = True)

## 10
df1, df2 = HT2.pop_seperator("DATASETS\Milk Transportation Cost Data.csv", by = "type")
HT2.two_sample(df1, df2, size = 0.10, equal_cov = False,  plot = True)

## 11
df1, df2, df3 = M.pop_seperator("DATASETS\Manova.csv",by = "pop")
M.test(df1, df2, df3, manova_table = True, size = 0.01 , plot = True)
