import csv
import os
import glob


with open('test.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)

	st1_folder = "./test_stg1"
	st2_folder = "./test_stg2"

	st1_list = glob.glob(os.path.join(st1_folder, "*"))
	st2_list = glob.glob(os.path.join(st2_folder, "*"))


	writer.writerow(["img_id"])


	for i in st1_list:
		writer.writerow([i])

	for j in st2_list:
		writer.writerow([j])

