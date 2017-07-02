
## Author - Himanshu Goyal ##

from pyspark import SparkContext, SparkConf, SQLContext
from datetime import datetime
from pyspark.sql import Row
import numpy as np
from itertools import groupby
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from operator import add
from pyspark import RDD
import random


conf = SparkConf().setAppName('PaytmLabs')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
Window_Size = 5

########################################         DATA ANALYTICS PART         #################################

def parse_log_file(data):
	split_data= data.split('\"')
	date = split_data[0].split()[0]
	ip = split_data[0].split()[2]
	url = split_data[1].split()[1]
	return ((date),(ip,url))

def __datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')

def get_seconds(t1,t2):
	return (t1-t2).total_seconds()

def ip_to_sessionize(data):
	ip,time_url_list = data
	st=0
	session_urls = []
	sessions=[]
	pr=0
	count=0
	for (time,url) in time_url_list:
		if(count==0):
			pr=time
			st=time
		if(get_seconds(__datetime(time),__datetime(pr))<=900):
			session_urls.append(url)
		else:
			session_time = get_seconds(__datetime(pr),__datetime(st))
			sessions.append((session_time,len(set(session_urls)),st,pr))
			st = time 
			session_urls = [url]

		if count==len(time_url_list)-1:
			session_time = get_seconds(__datetime(time),__datetime(st))
			session_urls.append(url)
			sessions.append((session_time,len(set(session_urls)),st,time))
		pr = time
		count+=1

	return (ip,sessions) ## ip -> user ip ; sessions -> (session_time,number of unique urls,session_start,session_end) ##

def average_session_time(data):

	ip,session_params = data
	total_session_time=0
	for t in session_params:
		total_session_time=total_session_time+t[0]

	return (total_session_time/len(session_params))

#####################################          MACHINE LEARNING PART        #######################################

def trend_value(tv):      ## used to calculate the trends for last 5 minutes
	y = tv
	y_mean = np.mean(y)
	x = [i for i in range(1,Window_Size+1)]
	x_mean = np.mean(x)

	return sum((x-x_mean)*(y-y_mean))/sum((x-x_mean)*(x-x_mean))

def mean_value(mv):    ## used to calculate the mean values for last 5 minutes
	return np.mean(mv)

def regression_data(data):
	see = data.split("\"")
	date = see[0].split()[0]
	hour = date.split(":")[0][-2:]
	minute = date.split(':')[1]
	return (hour,minute)

def train_test_dataset_Regression(m_l):

	mean_list=[]
	trend_list=[]
	for i in range(len(m_l)-Window_Size+1):
		mean_list.append(mean_value(m_l[i:i+Window_Size])) ## Mean values of last 5(Window Size) minutes  ##
		trend_list.append(trend_value(m_l[i:i+Window_Size])) ## Trend of last 5(window Size) minutes ##

	ws = Window_Size
	features=[]
	output = []
	for i in range(len(mean_list)-1):
		features.append([mean_list[i],trend_list[i],m_l[i+ws-4],m_l[i+ws-3],m_l[i+ws-2],m_l[i+ws-1]]) ## Features list for training and testing ##
		output.append(m_l[i+Window_Size])	## Outputs list for traning and testing ##

	index = random.sample(range(0, len(mean_list)-1), len(mean_list)-1) ## Random Sampling of train and test set
	train_index = index[:70]
	test_index = index[-35:]
	
	train_index = sorted(train_index)
	test_index = sorted(test_index)
	features = np.asarray(features)
	output = np.asarray(output)
	
	train_features =  features[train_index] 
	train_output = output[train_index]  
	test_features = features[test_index]
	test_output = output[test_index]

	return train_features,train_output,test_features,test_output

def plot_plot(pred,ty):

	plt.plot(pred,label="predictions")
	plt.plot(ty,label="actual")
	plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
	plt.show()


def SVR_Regression(x,y,tx,ty):

	clf = SVR(C=.001,gamma=0.001,kernel="linear")
	clf.fit(x,y)
	pred =clf.predict(tx)
	plot_plot(pred,ty)

	return pred

def Linear_Regression(x,y,tx,ty):

	LR = LinearRegression()
	LR.fit(x,y)
	pred = LR.predict(tx)
	plot_plot(pred,ty)

	return pred

###########     ###########     ###########     ###########     ###########     ###########     ###########     ###########     ###########     


if __name__ == "__main__":

	user_rdd = sc.textFile("2015_07_22_mktplace_shop_web_log_sample.log")

	parsed_data = user_rdd.map(parse_log_file) 
	ip_time_url = parsed_data.sortByKey().map(lambda line: (line[1][0], (line[0], line[1][1])))
	sessionized_data = ip_time_url.groupByKey().map(ip_to_sessionize) ## Sessionized data as per the IPs ##

	averaged_session_time = sessionized_data.map(average_session_time) ## Average session times (IP wise) in seconds ##
	avg_times = averaged_session_time.collect()
	sum_times=0
	for i in range(len(avg_times)):
		sum_times+=avg_times[i]

	avg_time = sum_times/len(avg_times) ## Average time of all the combined sessions during that day in seconds ##

	session_ip_rdd = sessionized_data.flatMap(lambda line: [(sess_t[0], line[0]) for sess_t in line[1]])
	sorted_session = session_ip_rdd.sortByKey(False) ## Sessions Sorted in Decreasing order as per Session Time ##

	
	request_per_minute = user_rdd.map(regression_data).sortBy(lambda line: (line[0],line[1])) \
		.map(lambda line : (line[0]+line[1],1)).reduceByKey(lambda a, b: a + b).sortByKey() ## gives the number of requests per minutes sorted by time

	list_request_per_minute = [x[1] for x in request_per_minute.toLocalIterator()]  
	train_x,train_y,test_x,test_y = train_test_dataset_Regression(list_request_per_minute)

	LR_Predictions = Linear_Regression(train_x,train_y,test_x,test_y)
	SVR_Predictions = SVR_Regression(train_x,train_y,test_x,test_y)
	