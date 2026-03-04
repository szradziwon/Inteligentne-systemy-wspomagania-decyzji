import numpy as np
import pymcdm.methods as mcdm
import matplotlib.pyplot as plt
'''
min max normalization implementation
'''
def MinMaxNormalize(matrix):
	matrix = matrix.astype(float)
	min_vals = matrix.min(axis=0)
	max_vals = matrix.max(axis=0)

	normalized = (matrix - min_vals) / (max_vals - min_vals)
	return normalized

'''
vector normalization implementation
'''
def VectorNormalization(matrix):
	matrix = matrix.astype(float)
	norm = matrix / np.sqrt((matrix ** 2).sum(axis=0))
	return norm

'''
linear normalization implementation
'''
def LinearNormalize(matrix, criteria_types):

	matrix = matrix.astype(float)
	normalized = np.zeros(matrix.shape)

	for j in range(matrix.shape[1]):

		if criteria_types[j] == 1:
			max_val = matrix[:, j].max()
			normalized[:, j] = matrix[:, j] / max_val

		else:
			min_val = matrix[:, j].min()
			normalized[:, j] = np.divide(
				min_val,
				matrix[:, j],
				out=np.zeros_like(matrix[:, j], dtype=float),
				where=matrix[:, j] != 0
			)

	return normalized

'''
My topsis implementation
'''
def topsis(matrix, weights, criteriaTypes, normalization_type = "MinMax"):
	if normalization_type == "MinMax":
		norm = MinMaxNormalize(matrix)
	elif normalization_type == "VectorNormalization":
		norm = VectorNormalization(matrix)
	elif normalization_type == "LinearNormalize":
		norm = LinearNormalize(matrix, criteriaTypes)
	else:
		norm = MinMaxNormalize(matrix)

	weighted = norm * weights

	ideal = np.zeros(matrix.shape[1])
	anti = np.zeros(matrix.shape[1])

	for j in range(matrix.shape[1]):

		if criteriaTypes[j] == 1:     # benefit
			ideal[j] = weighted[:, j].max()
			anti[j] = weighted[:, j].min()
		else:                          # cost
			ideal[j] = weighted[:, j].min()
			anti[j] = weighted[:, j].max()
	#euk distance
	distIdeal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
	distAnti = np.sqrt(((weighted - anti) ** 2).sum(axis=1))

	score = distAnti / (distIdeal + distAnti)

	return score

def PrintMatrixWithHeaders(matrix, headers):
	col_width = 18
	for h in headers:
		print(f"{h:<{col_width}}", end="")
	print()
	for row in matrix:
		for val in row:
			print(f"{val:<{col_width}.2f}", end="")
		print()

def plot_scores(scores, title):
	x = np.arange(len(scores))

	plt.figure()
	plt.bar(x, scores)
	plt.xticks(x, [f"Car {i}" for i in x], rotation=45, ha="right")
	plt.ylabel("TOPSIS score")
	plt.title(title)
	plt.tight_layout()
	plt.show()


#initial data
columnHeaders = ["Price", "Service cost", "Age", "HP", "Fuel consumption", "trunk capacity"]
criteriaWeights = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
criteriaTypes = np.array([-1, -1, -1, 1, -1, 1])

carParametersMatrix = np.array([[100000, 1500, 0, 120, 5.5, 600],
								 [120000, 3500, 5, 150, 4.5, 650],
								 [40000, 10000, 10,180,   8, 550],
								 [80000,  2500, 2, 110,   6, 400],
								 [20000,  2000,10,	60, 5.5, 250],
								 [35000,  2300, 4,  70,  6.5, 350],
								 [65000,  5500, 7, 100,   7, 450],
								 [95000,  1750, 1, 115, 5.5, 600]])
print("Basic matrix:")
PrintMatrixWithHeaders(carParametersMatrix, columnHeaders)

#normalized matrix
print("Min-max normalized:")
PrintMatrixWithHeaders(MinMaxNormalize(carParametersMatrix), columnHeaders)

print("\nVector normalized:")
PrintMatrixWithHeaders(VectorNormalization(carParametersMatrix), columnHeaders)

print("\n Linear normalized:")
PrintMatrixWithHeaders(LinearNormalize(carParametersMatrix, criteriaTypes), columnHeaders)

#topsis w/ MinMax Normalization
minMaxScores = topsis(carParametersMatrix, criteriaWeights, criteriaTypes, normalization_type = "MinMax")
minMaxRanking = np.argsort(minMaxScores)[::-1]
print()
print("scores:", minMaxScores)
print("ranking:", minMaxRanking)

print("\nTOPSIS, MinMax normalization ranking:")
for i, idx in enumerate(minMaxRanking, start=1):
	print(f"{i}. Car {idx} -> score = {minMaxScores[idx]:.4f}")
plot_scores(minMaxScores, "TOPSIS scores - MinMax normalization")

#topsis w/ vector Normalization
vectorScores = topsis(carParametersMatrix, criteriaWeights, criteriaTypes, normalization_type = "VectorNormalization")
vectorRanking = np.argsort(vectorScores)[::-1]
print()
print("scores:", vectorScores)
print("ranking:", vectorRanking)

print("TOPSIS, vector normalization ranking:")
for i, idx in enumerate(vectorRanking, start=1):
	print(f"{i}. Car {idx} -> score = {vectorScores[idx]:.4f}")
plot_scores(vectorScores, "TOPSIS scores - Vector normalization")

#topsis w/ Linear Normalization
linearScores = topsis(carParametersMatrix, criteriaWeights, criteriaTypes, normalization_type = "LinearNormalize")
linearRanking = np.argsort(linearScores)[::-1]
print()
print("Linear scores:", linearScores)
print("Linear ranking:", linearRanking)

print("\nTOPSIS, linear normalization ranking:")
for i, idx in enumerate(linearRanking, start=1):
	print(f"{i}. Car {idx} -> score = {linearScores[idx]:.4f}")
plot_scores(linearScores, "TOPSIS scores - Linear normalization")

#topsis built-in pymcdm
topsis_mcdm = mcdm.TOPSIS()
scoresMCDM = topsis_mcdm(carParametersMatrix, criteriaWeights, criteriaTypes)
MCDMRanking = np.argsort(scoresMCDM)[::-1]
print()
print("TOPSIS scores:", scoresMCDM)
print("TOPSIS ranking:", MCDMRanking)
print("\nTOPSIS MCDM ranking:")
for i, idx in enumerate(MCDMRanking, start=1):
	print(f"{i}. Car {idx} -> score = {scoresMCDM[idx]:.4f}")
plot_scores(scoresMCDM, "TOPSIS scores - pymcdm")

#VIKOR method
vikor_mcdm = mcdm.VIKOR()
scoresVikor = vikor_mcdm(carParametersMatrix, criteriaWeights, criteriaTypes)
vikorRanking = np.argsort(scoresVikor)[::-1]
print()
print("VIKOR scores:", scoresVikor)
print("VIKOR ranking:", vikorRanking)
for i, idx in enumerate(vikorRanking, start=1):
	print(f"{i}. Car {idx} -> score = {scoresVikor[idx]:.4f}")
plot_scores(scoresVikor, "Vikor scores - pymcdm")

#Promethee II
prometheeII = mcdm.PROMETHEE_II('usual')
scoresPromethee = prometheeII(carParametersMatrix, criteriaWeights, criteriaTypes)
prometheeRanking = np.argsort(scoresPromethee)[::-1]
print()
print("PROMETHEE scores:", scoresPromethee)
print("PROMETHEE ranking:", prometheeRanking)
for i, idx in enumerate(prometheeRanking, start=1):
	print(f"{i}. Car {idx} -> score = {scoresPromethee[idx]:.4f}")
plot_scores(scoresPromethee, "PROMETHEE scores - pymcdm")