#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project:  (Medical Appointment No Shows)
# # Hessa Alqahtani
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# > This is an attempt to investigate Medical Appointment No Shows dataset of 110527 patient appointments in Brazil from late April to early June 2016. A number of characteristics about each patient are included in each row. Data shows that 30% patients miss thier appointments. The data sets is provided on Kaggle.
#  My goal is trying to answer the questions by investigating the dataset.\
# **Questions:**
# > - Who is more committed to attending the appointment, female or male ?
# > - Are no-show appointments associated with a certain gender?
# > - Do the SMS reminders decrease the number of absences?
# > - Are illnesses like Handicap affecting by patient's age?
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# In[2]:


df = pd.read_csv('noshowappointments.csv',                        parse_dates=['ScheduledDay', 'AppointmentDay'])
df.head()


# > I used **head()** to display the first five rows of the data.

# In[3]:


df.info()


# In[4]:


#this function takes data frame (df) as a parameter and prints out the number of rows and columns.
def shape(df):
    print('Number of columns is',len(df.columns))
    print('Number of rows is',len(df.index))


# In[5]:


shape(df)


# In[6]:


df.info()


# > After i loaded my data, I found out that the type of "ScheduledDay and AppointmentDay" was object while it's supposed to be datetime. So, I used the argument "parse_dates" to change it's type.

# In[7]:


df.describe()


# > **describe()** function help me to look at basic statistics of the numerical features in my data.

# In[8]:


df.nunique()


# > I used **unique()** function to get the unique values in my data.

# 
# 
# ### Data Cleaning (the process of removing incorrect, corrupted, incorrectly formatted, duplicate, or missing values)

# In[9]:


df.duplicated().sum()


# In[10]:


df.isnull().sum()


# > As shown above, The data does not have any missing values nor duplicate.

# In[11]:


df.drop('PatientId',axis=1, inplace=True)


# In[12]:


df.drop('AppointmentID',axis=1, inplace=True)


# > Here, I removed some columns that were not necessary to make the data processing faster.

# In[13]:


df= df.rename(columns={'No-show': 'miss_their_appointments'})


# > I renamed “No-show” column to “miss_their_appointments” to help me avoid any confusion. First time i saw the data, I thought that the "No" values mean that they missed their appointments while it's the opposite.

# In[14]:


df=df.rename(columns={'Handcap':'Handicap'})


# > I renamed misspelled column name.

# In[15]:


df.head()


# > i called the head() again just to check the data after the clean process.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
#  
# 
# ### Research Question 1 (Is the gender going to affect  the appointments?)

# In[16]:


df.Gender.value_counts()


# In[17]:


df.Gender.unique()


# In[18]:


df.groupby('Gender').size().plot(kind='pie' ,autopct='%.2f', colors = ['pink', 'lightblue']);
plt.title("Gender");
plt.ylabel('pie chart');


# > As shown above, Female is the greater proportion, They takes way more care of their health in comparison to man.
# In total 71840 F against 38687 M.
# > 1. I checked the value counts and unique value in Gender column.
# > 2. I used pie chart for this data because it's single-variable **1D** explorations, I learned how to use autopct in this chart.

# In[19]:


df['miss_their_appointments'].value_counts()


# In[20]:


df.miss_their_appointments.unique()


# In[21]:


df_gender=df.groupby('Gender').miss_their_appointments.value_counts()


# In[22]:


df_gender.unstack().plot.bar(color=['pink','lightblue'])
plt.title("Is the Gender going to affect the appointments?");
plt.ylabel('patients');
plt.xlabel('Gender');


# > As shown above, +88K appeared in their appointments, 57k were female and 30k male. And 22k miss their appointments were 13k females and 7k males. So, After i calculated the ratio for female (57:13 =4.38) and male(30:7=4.28) as we can see it's almost the same ratio for both who came to their appointments and who didn't. As a result Gender doesn't affect the appointments.
# > 1. As noted previously, We need to check the value count and unique value for the columns needed.
# > 2.  This time I used "GroupBy" function. It allows me to split my data into separate groups to perform computations for better analysis.
# > 3. I used bar chart because i want to compare more than one column and it helps to show the comparison values clearly.

# ### Research Question 2  (Dose the SMS help the patients to avoid missing their appointments? )

# In[23]:


df.SMS_received.value_counts()


# In[24]:


df_sms=df.groupby('SMS_received').miss_their_appointments.value_counts()


# In[25]:


df_sms.unstack().plot.bar(color=['pink','lightblue'])
plt.title("Is the SMS going to affect the appointments?")
plt.ylabel('patients');
plt.xlabel('messages sent to the patient');


# > in this bar chart i took two columns (SMS_received and miss_their_appointments) to see if the SMS will help patients to attend their appointments. As shown in the bar, +60k came and around 11k miss their appointments without the sms received. And from total people received messages (35482) around 27k came to their appointment after the message! and only 10k missed their appointments which show that sms messages really help the patient to remember his appointments.

# ### Research Question 3 (Relation between Age and handicap )

# In[26]:


df.Handicap.value_counts()


# In[27]:


df.Handicap.unique()


# In[28]:


df.plot.scatter(x="Handicap", y="Age", color=['lightblue']);


# >**Add at least one scatter plot or correlation matrix to show relationships between variables**
# i decided to add the two(Age,Handicap) in scatter to show the relationships between variables.
# As shown above there are 108286 patients doesn't suffer from any handicap. and then it drops to 2042 patients with level 1 handicap(serious difficulty walking or climbing stairs). 
# I took "age" varible  because it's the only one that have a better distribution between the amount of data.

# <a id='conclusions'></a>
# ## Conclusions
# > **Results**:\
# 1- Female is the greater proportion in this dataset with 65% compaere to 35% M.\
# 2- The gender doesn't affect the appointments.\
# 3- There is a higher percentage of patients who recevied SMS and didn't show up when compared to patients who recevied SMS and did show up. Which means SMS reminders help the patients to come to their appointments.
# 
# **Limitations**:
# >- The data set explored in this analysis was just over a 6 month period. 
# >- Patients with repeat "miss_their_appointments" would be a good cohort to remove from the analysis to understand environmental or health factors influencing their appointments.
# >- We don't have a lot of details to draw conclusions.
# > -We don't have a good distribution between the amount of data.
# ## Submitting your Project 
# 
# 
