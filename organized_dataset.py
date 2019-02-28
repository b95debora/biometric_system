import numpy as np
import glob
import shutil
import cv2
import random

age_range0 = range(0,3)
age_range1 = range(5,9)
age_range2 = range(11,16)
age_range3 = range(18,24)
age_range4 = range(26,33)
age_range5 = range(36,45)
age_range6 = range(48,55)
age_range7 = range(60,70)
age_range8 = range(73,116)

AGE = [age_range0, age_range1, age_range2, age_range3, age_range4, age_range5, age_range6, age_range7, age_range8]
GENDER = ["male", "female"]
RACE = ["white", "black", "asian", "indian", "others"]
#numer of images for each class
NUMB_IMAGES = 5000
IMG_SIZE = 224 #image size for cnn
TRAINING_ONECLASS_SIZE = 130
VALIDAZION_ONECLASS_SIZE = 30
TEST_ONECLASS_SIZE = 70

#create a 3-dimensional array: age_row, race_column, gender_depth, and each matrix element rapresents the images samples. 
#for istance: M[0,0,0] is the name list of images with AGE[0], RACE [0], GENDER[0]
#M[0,0,1] refers the images with gender[1] = female
age_race_gender_image = np.empty ((len(AGE), len(RACE), len(GENDER)), object)

def organized_dataset(folders):
	for folder in folders:
	    images = glob.glob(folder+"*")
	    for img_path in images:
	    	img = parse_path(img_path)
	    	age_index = age2index(int(img[0]))
	    	if (age_index != "pass" and len(img) == 3 and img[0]!="" and img[1]!="" and img[2]!=""):
		    	age = age_index
		    	gender = int(img[1])
		    	race = int(img[2])
		    	if age_race_gender_image[age,race,gender] == None:
		    		age_race_gender_image[age,race,gender] = []
		    	if (len(age_race_gender_image[age,race,gender]) < NUMB_IMAGES):
		    		age_race_gender_image[age,race,gender].append(img_path)

#given image_path, extract age, gender, race informatio. Example: input: part1\100_1_0_20170110183726390.jpg output: [100, 1, 0]
def parse_path(path):
	fields = path.split("_")
	age_field = fields[0].split("\\")[1]
	fields[0] = age_field
	#delete time field
	fields.pop()
	return fields
	

#return the index of age_range which age belongs to
def age2index(age):
	indice = 0
	while (indice < len(AGE) and age not in AGE[indice]):
		indice = indice + 1
	if (indice >= len(AGE)):
		return "pass"
	return indice

# Input: single cell of a multidimensional array (that cell is an array of image names with same gender race and age range)
# Output: age range, gender and race that characterizes the cell
def print_class(element):
	string = parse_path(element[0])
	age = int(string[0])
	gender = int(string[1])
	race = int(string[2])
	for age_range in AGE:
		if (age in age_range):
			age = str(age_range)
	gender = GENDER[gender]
	race = RACE[race]
	return [age, gender, race]

# Print informations about the cells of the multidimensional array and statistics about how much images and list are used and created
def print_lists(multidimensional_array):
	number_of_images = 0 # total images extracted from dataset counter
	number_of_iterations = 0 # created list counter
	for i in multidimensional_array.reshape(-1): # reshape(-1) unrolls the multidimensional_array
		print("Numero di immagini: "+str(np.size(i))+" Tipo di immagine: "+str(print_class(i)))
		number_of_images = number_of_images + np.size(i)
		number_of_iterations = number_of_iterations + 1
	print("total considered images = " + str(number_of_images) + " numero di liste = " + str(number_of_iterations))

#return three list of images names for training validation and test of a gender cnn, that list are balanced considering both age, gender and race
def split_into_training_validation_test_set_gender():
	training_size_for_one_class = TRAINING_ONECLASS_SIZE
	validation_size_for_one_class = VALIDAZION_ONECLASS_SIZE
	test_size_for_one_class = TEST_ONECLASS_SIZE
	classnumber = 2
	current_training_size = 0
	current_validation_size = 0
	current_test_size = 0
	training_set = []
	validation_set = []
	test_set = []
	temp_array = np.copy(age_race_gender_image)
	while(current_training_size<(training_size_for_one_class*classnumber)):
		for gender_iterator in range(len(GENDER)):
			for race_iterator in range(len (RACE)):
				for age_iterator in range(len(AGE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_training_size<(training_size_for_one_class*classnumber)) ):
						training_set.append(cell_list.pop())
						current_training_size=current_training_size+1
	while(current_validation_size<(validation_size_for_one_class*classnumber)):
		for gender_iterator in range(len(GENDER)):
			for race_iterator in range(len (RACE)):
				for age_iterator in range(len(AGE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_validation_size<(validation_size_for_one_class*classnumber)) ):
						validation_set.append(cell_list.pop())
						current_validation_size=current_validation_size+1
	while(current_test_size<(test_size_for_one_class*classnumber)):
		for gender_iterator in range(len(GENDER)):
			for race_iterator in range(len (RACE)):
				for age_iterator in range(len(AGE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_test_size<(test_size_for_one_class*classnumber)) ):
						test_set.append(cell_list.pop())
						current_test_size=current_test_size+1
	return training_set, validation_set, test_set

#return three list of images names for training validation and test of a race cnn, that list are balanced considering both age, gender and race
def split_into_training_validation_test_set_race():
	training_size_for_one_class = TRAINING_ONECLASS_SIZE
	validation_size_for_one_class = VALIDAZION_ONECLASS_SIZE
	test_size_for_one_class = TEST_ONECLASS_SIZE
	classnumber = 5
	current_training_size = 0
	current_validation_size = 0
	current_test_size = 0
	training_set = []
	validation_set = []
	test_set = []
	temp_array = np.copy(age_race_gender_image)
	while(current_training_size<(training_size_for_one_class*classnumber)):
		for race_iterator in range(len(RACE)):
			for gender_iterator in range(len (GENDER)):
				for age_iterator in range(len(AGE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_training_size<(training_size_for_one_class*classnumber)) ):
						training_set.append(cell_list.pop())
						current_training_size=current_training_size+1
	while(current_validation_size<(validation_size_for_one_class*classnumber)):
		for race_iterator in range(len(RACE)):
			for gender_iterator in range(len (GENDER)):
				for age_iterator in range(len(AGE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_validation_size<(validation_size_for_one_class*classnumber)) ):
						validation_set.append(cell_list.pop())
						current_validation_size=current_validation_size+1
	while(current_test_size<(test_size_for_one_class*classnumber)):
		for race_iterator in range(len(RACE)):
			for gender_iterator in range(len (GENDER)):
				for age_iterator in range(len(AGE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_test_size<(test_size_for_one_class*classnumber)) ):
						test_set.append(cell_list.pop())
						current_test_size=current_test_size+1
	return training_set, validation_set, test_set

#return three list of images names for training validation and test of a age cnn, that list are balanced considering both age, gender and race
def split_into_training_validation_test_set_age():
	training_size_for_one_class = TRAINING_ONECLASS_SIZE
	validation_size_for_one_class = VALIDAZION_ONECLASS_SIZE
	test_size_for_one_class = TEST_ONECLASS_SIZE
	classnumber = 9
	current_training_size = 0
	current_validation_size = 0
	current_test_size = 0
	training_set = []
	validation_set = []
	test_set = []
	temp_array = np.copy(age_race_gender_image)
	while(current_training_size<(training_size_for_one_class*classnumber)):
		for age_iterator in range(len(AGE)):
			for gender_iterator in range(len (GENDER)):
				for race_iterator in range(len(RACE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_training_size<(training_size_for_one_class*classnumber)) ):
						training_set.append(cell_list.pop())
						current_training_size=current_training_size+1
	while(current_validation_size<(validation_size_for_one_class*classnumber)):
		for age_iterator in range(len(AGE)):
			for gender_iterator in range(len (GENDER)):
				for race_iterator in range(len(RACE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_validation_size<(validation_size_for_one_class*classnumber)) ):
						validation_set.append(cell_list.pop())
						current_validation_size=current_validation_size+1
	while(current_test_size<(test_size_for_one_class*classnumber)):
		for age_iterator in range(len(AGE)):
			for gender_iterator in range(len (GENDER)):
				for race_iterator in range(len(RACE)):
					cell_list = temp_array[age_iterator,race_iterator,gender_iterator]
					if ((len(cell_list)!= 0) and (current_test_size<(test_size_for_one_class*classnumber)) ):
						test_set.append(cell_list.pop())
						current_test_size=current_test_size+1
	return training_set, validation_set, test_set

#create training_data validation_data and test_data
#input: the supercategory for which you want create the sets, must be gender, race or age
#outut: sets of images and labels
def create_train_validation_test(supercategory):
	part1 = "datasetsistemibiometrici\\"
	folders = [part1]
	organized_dataset(folders)
	training_data = []
	validation_data = []
	test_data = []
	if(supercategory=="gender"):
		trainnames, validationnames, testnames = split_into_training_validation_test_set_gender()
	elif(supercategory=="race"):
		trainnames, validationnames, testnames = split_into_training_validation_test_set_race()
	elif(supercategory=="age"):
		trainnames, validationnames, testnames = split_into_training_validation_test_set_age()
	else:
		print("error no such supercategory defined or errate supercategory")
	# create train
	for imagepath in trainnames:
		temp = parse_path(imagepath)
		if(supercategory=="gender"):
			class_num = int(temp[1])
		elif(supercategory=="race"):
			class_num = int(temp[2])
		elif(supercategory=="age"):
			class_num = int(temp[0])
		else:
			print("error no such supercategory defined or errate supercategory")
		try:
			img_array = cv2.imread(imagepath, cv2.IMREAD_COLOR)
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
			training_data.append([new_array, class_num])
		except Exception as e:
			pass
	#create validation
	for imagepath in validationnames:
		temp = parse_path(imagepath)
		if(supercategory=="gender"):
			class_num = int(temp[1])
		elif(supercategory=="race"):
			class_num = int(temp[2])
		elif(supercategory=="age"):
			class_num = int(temp[0])
		else:
			print("error no such supercategory defined or errate supercategory")
		try:
			img_array = cv2.imread(imagepath, cv2.IMREAD_COLOR)
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
			validation_data.append([new_array, class_num])
		except Exception as e:
			pass
	#create test
	for imagepath in testnames:
		temp = parse_path(imagepath)
		if(supercategory=="gender"):
			class_num = int(temp[1])
		elif(supercategory=="race"):
			class_num = int(temp[2])
		elif(supercategory=="age"):
			class_num = int(temp[0])
		else:
			print("error no such supercategory defined or errate supercategory")
		try:
			img_array = cv2.imread(imagepath, cv2.IMREAD_COLOR)
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
			test_data.append([new_array, class_num])
		except Exception as e:
			pass
	random.shuffle(training_data)
	random.shuffle(validation_data)
	random.shuffle(test_data)
	return training_data, validation_data, test_data

#def conta(a):
#	male = 0
#	female = 0
#	white = 0
#	black = 0
#	asian = 0
#	indian = 0
#	others = 0
#
#	for i in a: 
#		string = parse_path(i)
#		age = int(string[0])
#		gender = int(string[1])
#		race = int(string[2])
#		for age_range in AGE:
#			if (age in age_range):
#				age = str(age_range)
#		gender = GENDER[gender]
#		race = RACE[race]
#		if(gender=="male"):
#			male = male + 1
#		elif(gender=="female"):
#			female= female+1
#		else:
#			male = male
#		if(race=="white"):
#			white = white + 1
#		elif(race=="black"):
#			black= black+1
#		elif(race=="asian"):
#			asian= asian+1
#		elif(race=="indian"):
#			indian= indian+1
#		elif(race=="others"):
#			others= others+1
#		else:
#			white = white
#		print(" Tipo di immagine: "+str([age,race,gender]))
#	print (np.size(a))
#	print ("male = " + str(male))
#	print ("female = " + str(male))
#	print ("white = " + str(white))
#	print ("black = " + str(black))
#	print ("asian = " + str(asian))
#	print ("indian = " + str(indian))
#	print ("others = " + str(others))



#def main():
#	part1 = "datasetsistemibiometrici\\"
#	part2 = "part2\\"
#	part3 = "part3\\"
#	folders = [part1]#, part2, part3]
#	print("organizing dataset..")
#	organized_dataset(folders)
#	print("writing the check.txt file..")
	#check_file = "check.txt"
	#print_numpy_file(check_file, age_race_gender_image)
#	print_lists(age_race_gender_image)
#	a,b,c = create_train_validation_test("gender")
  	

#if __name__ == "__main__":
#	main()