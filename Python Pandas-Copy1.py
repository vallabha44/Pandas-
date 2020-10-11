#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
a = {'student-name':['bobby','Oliver','Harry','George','Jack','Leo'],
    '1st-sem':[45,56,47,67,35,53]}
df = pd.DataFrame(a)
print(df)

Output:
      student-name  1st-sem
0        bobby       45
1       Oliver       56
2        Harry       47
3       George       67
4         Jack       35
5          Leo       53

# In[3]:
df.shape
Output:
    (6, 2)


# In[4]:
df.index
Output: RangeIndex(start=0, stop=6, step=1)

# In[5]:
df.size
Output: 12

# In[6]:
df.columns
Output: Index(['student-name', '1st-sem'], dtype='object')

# In[7]:
df.values
Output: array([['bobby', 45],
       ['Oliver', 56],
       ['Harry', 47],
       ['George', 67],
       ['Jack', 35],
       ['Leo', 53]], dtype=object)

# In[8]:
df.axes
Output: [RangeIndex(start=0, stop=6, step=1),
 Index(['student-name', '1st-sem'], dtype='object')]

# In[9]:
df.ndim
Output: 2


# In[10]:
df['2nd-sem']=[34,36,45,67,63,55]
df
Output: 
student-name	1st-sem	2nd-sem
0	bobby	       45   	34
1	Oliver	       56       36
2	Harry	       47	45
3	George	       67	67
4	Jack	       35	63
5	Leo	       53	55


# In[11]: Renaming the columns
df.columns=['stdnt-name','first-sem','second-sem']
df
Output:	stdnt-name	first-sem	second-sem
0	    bobby	        45	        34
1	    Oliver	        56	        36
2	    Harry	        47	        45
3	    George	        67	        67
4	    Jack	        35	        63
5	    Leo	            	53	        55

# # Pandas Methods

# In[12]:
df.count()
Output: 
stdnt-name    6
first-sem     6
second-sem    6
dtype: int64

# In[13]:
df.sum()
Output:
stdnt-name    bobbyOliverHarryGeorgeJackLeo
first-sem                               303
second-sem                              300
dtype: object

    
# In[14]:
df.max()
Output:
stdnt-name    bobby
first-sem        67
second-sem       67
dtype: object

# In[15]:
df.describe()
Output:
        first-sem	second-sem
count       6.00000	6.000000
mean	    50.50000	50.000000
std	    10.87658	13.856406
min	    35.00000	34.000000
25%	    45.50000	38.250000
50%	    50.00000	50.000000
75%	    55.25000	61.000000
max	    67.00000	67.000000

# In[16]:
df.head()
Output:
	stdnt-name	first-sem	second-sem
0	bobby	        45	        34
1	Oliver	        56	        36
2	Harry	        47	        45
3	George	        67	        67
4	Jack	        35	        63

# In[17]:
df.head(3)
Output:
	stdnt-name	first-sem	second-sem
0	bobby	        45	        34
1	Oliver	        56	        36
2	Harry	        47	        45


# In[18]:
df.tail()
Output:
	stdnt-name	first-sem	second-sem
1	Oliver	        56	        36
2	Harry	        47	        45
3	George	        67	        67
4	Jack	        35	        63
5	Leo	        53	        55

# In[19]:
df.tail(4)
Output:
	stdnt-name	first-sem	second-sem
2	Harry	        47	        45
3	George	        67	        67
4	Jack	        35	        63
5	Leo	        53	        55

# In[20]:
df.mean()
Output:
first-sem     50.5
second-sem    50.0
dtype: float64

# In[21]:
df.std()
Output:
first-sem     10.876580
second-sem    13.856406
dtype: float64

# In[22]:
df.sample()
Output:
   	stdnt-name	first-sem	second-sem
3	George	        67	        67


# In[23]:Displays 2rows randomly 
df.sample(n=2)
Output:
    stdnt-name	first-sem	second-sem
3	George	        67	        67
5	Leo	        53	        55

# In[24]: Displays 5 rows randomly
df.sample(n=5)
Output:
    stdnt-name	first-sem	second-sem
3	George	        67	       67
2	Harry	        47	       45
4	Jack	        35	       63
5	Leo	        53	       55
1	Oliver	        56	       36

# In[25]:
df.median()
Output:
first-sem     50.0
second-sem    50.0
dtype: float64

# In[26]:
df.sort_values(by='first-sem',ascending=False)
Output:
    stdnt-name	first-sem	second-sem
3	George	        67	        67
1	Oliver	        56	        36
5	Leo	        53	        55
2	Harry	        47	        45
0	bobby	        45	        34
4	Jack	        35	        63

# In[27]:
df.sort_values(by='first-sem')
Output:
    stdnt-name	first-sem	second-sem
4	Jack	        35	        63
0	bobby	        45	        34
2	Harry	        47	        45
5	Leo	        53	        55
1	Oliver	        56	        36
3	George	        67	        67
    

# In[28]:
df.loc[0]
Output:
 stdnt-name    bobby
first-sem        45
second-sem       34
Name: 0, dtype: object

# In[29]:
df.loc[5]
Output:
 stdnt-name    Leo
first-sem      53
second-sem     55
Name: 5, dtype: object


# In[30]:
df.iloc[:,-1]
Output:
0    34
1    36
2    45
3    67
4    63
5    55
Name: second-sem, dtype: int64


# In[31]:
df.iloc[:,:]
Output:
   stdnt-name	first-sem	second-sem
0	bobby	        45	        34
1	Oliver	        56	        36
2	Harry	        47	        45
3	George	        67	        67
4	Jack	        35	        63
5	Leo	        53	        55

# In[32]:Creating a new column
import numpy as np
df['third-sem']=[34,56,np.nan,35,55,47]
df
Output:
    stdnt-name	first-sem	second-sem	third-sem
0	bobby	        45	        34	        34.0
1	Oliver	        56	        36	        56.0
2	Harry	        47	        45	        NaN
3	George	        67	        67	        35.0
4	Jack	        35	        63	        55.0
5	Leo	        53	        55	        47.0

# In[33]:
df.isna()
Output:
stdnt-name	first-sem	second-sem	  third-sem
0	False	    False	    False	   False
1	False	    False	    False	   False
2	False	    False	    False	   True
3	False	    False	    False	   False
4	False	    False	    False	   False
5	False	    False	    False	   False

# In[34]:
df.notnull()
Output:
	stdnt-name	first-sem second-sem	third-sem
0	True	        True	    True	    True
1	True	        True	    True	    True
2	True	        True	    True	    False
3	True	        True	    True	    True
4	True	        True	    True	    True
5	True	        True	    True	    True

# In[35]:
df.fillna(45)
Output:
    stdnt-name	first-sem	second-sem	third-sem
0	bobby	        45	        34	        34.0
1	Oliver	        56	        36	        56.0
2	Harry	        47	        45	        45.0
3	George	        67	        67	        35.0
4	Jack	        35	        63	        55.0
5	Leo	        53	        55	        47.0

# In[36]:Dropping specified row
df.drop(3)
Output:
	stdnt-name	first-sem	second-sem	third-sem
0	bobby	        45	        34	        34.0
1	Oliver	        56	        36	        56.0
2	Harry	        47	        45	        NaN
4	Jack	        35	        63	        55.0
5	Leo	        53	        55	        47.0

# In[37]:Drops the row which is having an empty value in dataset
df.dropna()
Output:
	stdnt-name	first-sem	second-sem	third-sem
0	bobby	        45	        34	        34.0
1	Oliver	        56	        36	        56.0
3	George	        67	        67	        35.0
4	Jack	        35	        63	        55.0
5	Leo	        53	        55	        47.0


# In[38]:Drops the column which is having an empty values in dataset
df.dropna(axis=1,how='any')
Output: 
student-name	firstt-sem	second-sem
0	bobby	       45   	34
1	Oliver	       56       36
2	Harry	       47	45
3	George	       67	67
4	Jack	       35	63
5	Leo	       53	55


# In[39]:
df.dropna(axis=0,how='any')
Output:
	stdnt-name	first-sem	second-sem	third-sem
0	bobby	        45	        34	        34.0
1	Oliver	        56	        36	        56.0
3	George	        67	        67	        35.0
4	Jack	        35	        63	        55.0
5	Leo	        53	        55	        47.0


# In[40]: It will work only when all the values in a row are 'NaN'
df.dropna(axis=0,how='all')
Output:
	stdnt-name	first-sem	second-sem	third-sem
0	bobby	        45	        34	        34.0
1	Oliver	        56	        36	        56.0
2	Harry	        47	        45	        NaN
4	Jack	        35	        63	        55.0
5	Leo	        53	        55	        47.0


# In[114]:
x = {'stdnt-name':['bobby','Oliver','Harry','George','Jack','Leo'],
    'first-sem':[45,56,47,67,35,53],
    'second-sem':[37,46,70,51,49,54],
    'third-sem':[51,43,61,45,37,62]}
y = {'stdnt-name':['charlie','butcher','hughie','MM','Kimiko','Homelander'],
    'first-sem':[35,46,57,59,68,39],
    'second-sem':[37,46,70,51,49,54],
    'third-sem':[51,43,61,45,37,62]}
df = pd.DataFrame(x,index=[0,1,2,3,4,5])
de = pd.DataFrame(y,index=[6,7,8,9,10,11])
c = pd.concat([df,de])
c
Output:
	stdnt-name	first-sem	second-sem	third-sem
0	bobby	        45	        37	        51
1	Oliver	        56	        46	        43
2	Harry	        47	        70	        61
3	George	        67	        51	        45
4	Jack	        35	        49	        37
5	Leo	        53	        54	        62
6	charlie	        35	        37	        51
7	butcher	        46	        46	        43
8	hughie	        57	        70	        61
9	MM	        59	        51	        45
10	Kimiko	        68	        49	        37
11	Homelander	39	        54	        62

# In[115]:
result = df.append(de)
result
Output:
	stdnt-name	first-sem	second-sem	third-sem
0	bobby	        45	        37	        51
1	Oliver	        56	        46	        43
2	Harry	        47	        70	        61
3	George	        67	        51	        45
4	Jack	        35	        49	        37
5	Leo	        53	        54	        62
6	charlie	        35	        37	        51
7	butcher	        46	        46	        43
8	hughie	        57	        70	        61
9	MM	        59	        51	        45
10	Kimiko	        68	        49	        37
11	Homelander	39	        54	        62



# In[116]:
result.groupby(['first-sem','second-sem']).groups
Output:
    {(35, 37): [6], (35, 49): [4], (39, 54): [11], (45, 37): [0], (46, 46): [7], (47, 70):
     [2], (53, 54): [5], (56, 46): [1], (57, 70): [8], (59, 51): [9], (67, 51): [3], (68, 49): [10]}

    
# In[117]:
result.insert(4,'fourth-sem',[51,43,61,45,37,62,51,43,61,45,37,62])
result
Output:
	stdnt-name	first-sem	second-sem	third-sem	fourth-sem
0	bobby	        45	        37	        51	        51
1	Oliver	        56	        46	        43	        43
2	Harry	        47	        70	        61	        61
3	George	        67	        51	        45	        45
4	Jack	        35	        49	        37	        37
5	Leo	        53	        54	        62	        62
6	charlie	        35	        37	        51	        51
7	butcher	        46	        46	        43	        43
8	hughie	        57	        70	        61	        61
9	MM	        59	        51	        45	        45
10	Kimiko	        68	        49	        37	        37
11	Homelander	39	        54	        62	        62


# In[118]:
result.insert(5,'fifth-sem',[51,43,61,45,37,62,51,43,61,45,37,62])
result
Output:
	stdnt-name	first-sem	second-sem	third-sem	fourth-sem	fifth-sem
0	bobby	        45	        37	        51	        51	        51
1	Oliver	        56	        46	        43	        43	        43
2	Harry	        47	        70	        61	        61	        61
3	George	        67	        51	        45	        45	        45
4	Jack	        35	        49	        37	        37	        37
5	Leo	        53	        54	        62	        62	        62
6	charlie	        35	        37	        51	        51	        51
7	butcher	        46	        46	        43	        43	        43
8	hughie	        57	        70	        61	        61	        61
9	MM	        59	        51	        45	        45	        45
10	Kimiko	        68	        49	        37	        37	        37
11	Homelander	39	        54	        62	        62	        62
    
# In[122]:
result.rename(columns={'fifth-sem':'5th-sem'})
Output:
	stdnt-name	first-sem	second-sem	third-sem	fourth-sem	5th-sem
0	bobby	        45	        37	        51	        51	        51
1	Oliver	        56	        46	        43	        43	        43
2	Harry	        47	        70	        61	        61	        61
3	George	        67	        51	        45	        45	        45
4	Jack	        35	        49	        37	        37	        37
5	Leo	        53	        54	        62	        62	        62
6	charlie	        35	        37	        51	        51	        51
7	butcher	        46	        46	        43	        43	        43
8	hughie	        57	        70	        61	        61	        61
9	MM	        59	        51	        45	        45	        45
10	Kimiko	        68	        49	        37	        37	        37
11	Homelander	39	        54	        62	        62	        62
    
    
# In[123]:
result.assign(sixthsem=[45,56,47,67,35,53,45,56,47,67,35,53],
             seventhsem=[45,56,47,67,35,53,45,56,47,67,35,53],
             eighthsem=[45,56,47,67,35,53,45,56,47,67,35,53])
Output:
	stdnt-name	first-sem	second-sem	third-sem	fourth-sem	fifth-sem	sixthsem	seventhsem	eighthsem
0	bobby	        45	        37	        51	        51	        51	        45	        45	        45
1	Oliver	        56	        46	        43	        43	        43	        56	        56	        56
2	Harry	        47	        70	        61	        61	        61	        47	        47	        47
3	George	        67	        51	        45	        45	        45	        67	        67	        67
4	Jack	        35	        49	        37	        37	        37	        35	        35	        35
5	Leo	        53	        54	        62	        62	        62	        53	        53	        53
6	charlie	        35	        37	        51	        51	        51	        45	        45	        45
7	butcher	        46	        46	        43	        43	        43	        56	        56	        56
8	hughie	        57	        70	        61	        61	        61	        47	        47	        47
9	MM	        59	        51	        45	        45	        45	        67	        67	        67
10	Kimiko	        68	        49	        37	        37	        37	        35	        35	        35
11	Homelander	39	        54	        62	        62	        62	        53	        53	        53    

# In[121]:
result.groupby(['third-sem']).agg([min,max])
Output:
	stdnt-name	first-sem	second-sem	fourth-sem	fifth-sem
       min	max	min	max	    min	max	    min	max	    min	max
third-sem										
37	Jack	Kimiko	35	68	    49	49	     37	37	    37	37
43	Oliver	butcher	46	56	    46	46	     43	43	    43	43
45	George	MM	59	67	    51	51	     45	45	    45	45
51	bobby	charlie	35	45	    37	37	     51	51	    51	51
61	Harry	hughie	47	57	    70	70	     61	61	    61	61
62	Homelander Leo	39	53	    54	54	     62	62	    62	62





