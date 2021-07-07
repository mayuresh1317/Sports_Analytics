#!/usr/bin/env python
# coding: utf-8

# In[5]:


#For writing and reading data into MongoDb as Collections from json files and carrying out various operations
import json
from pymongo import MongoClient
import pandas as pd
import numpy as np
import pprint

#For writing and reading data into PostgreSql
import psycopg2 as pg
import sqlalchemy as sqla

#For combinig all dataframes into one dataframe 
from functools import reduce as rd

# For Visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import seaborn as sns
import matplotlib.pyplot as plt
import random
import plotly.express as px
import plotly as py
import plotly.graph_objs as go


# # Part 1 - Reading and Writing to MongoDb and Data Pre-processing

# We have Data for PES Game players into 4 datasets namely - Player Details, Player Positions, Attacker & Midfielder attributes, and Defender and Goalkeeper attributes. 
# 
# Dataset worked on by
# 
# Player Details - Pooja Rakate;
# 
# Player Positions - Mayuresh Londhe;
# 
# Attacker & Midfielder attributes (AM Attr) - Ankit Rungta;
# 
# Defender and Goalkeeper attributes (DG Attr) - Priyanka Chimthankar

# In[2]:


##Establishing connection with MongoDb
#Port number: 27017

client = MongoClient('localhost', 27017)

#Database Name : PES_Players_20
mydb = client['PES_Players_20']

##Loading json files as collections into PES_Players_20 in MongoDb
try:
    #Added by Pooja Rakate
    #Creating a collection Player_Details by loading json file
    Player_Details = mydb.Player_Details
    with open('D:/Pooja_NCI/1_Semester/DataBase and Analytical programming/Project/Datasets/PES 2020/Player_Details.json', encoding = 'utf-8') as f:
        file_data = json.load(f)
        
    #Insert all the records from json file as documents in the collection
    Player_Details.insert_many(file_data)

    ##Added by Mayuresh Londhe
    #Creating a collection Player_Position_Ranking by loading json file
    PPRANK = mydb['Player_Position_Ranking']
    with open('D:/Pooja_NCI/1_Semester/DataBase and Analytical programming/Project/Code from everyone/Player_positions.json',encoding='utf-8') as f:
        file_data1 = json.load(f)

    #Insert all the records from json file as documents in the collection
    PPRANK.insert_many(file_data1)

    ##Added by Ankit Rungta
    #Creating a collection AM_Player_attributes by loading json file
    AM_Attributes = mydb['AM_Player_attributes']
    with open('D:/Pooja_NCI/1_Semester/DataBase and Analytical programming/Project/Code from everyone/AM_Player_attributes.json',encoding="utf-8") as f2:
        file_data2 = json.load(f2)
    
     #Insert all the records from json file as documents in the collection
    AM_Attributes.insert_many(file_data2)

    ##Added by Priyanka Chimthankar
    #Creating a collection DG_Players_Attributes by loading json file
    DG_Attributes = mydb['DG_Players_Attributes']
    with open('D:/Pooja_NCI/1_Semester/DataBase and Analytical programming/Project/Code from everyone/DG_Player_attributes.json',encoding='utf-8') as f3:
        file_data3 = json.load(f3)
    
     #Insert all the records from json file as documents in the collection
    DG_Attributes.insert_many(file_data3)
    
except:
    print('Table not loaded into MongoDB !!')


# ##### Start of code by Pooja Rakate for Data pre-processing of Player Details data

# In[178]:


##Checking if collection is created properly in MongoDb by accessing one document from it
Player_Details.find_one()


# In[3]:


## Printing all the documents from collection Player_Details. This works like the query "select * from Player_details"
for record in Player_Details.find():
    print(record)


# In[73]:


##reading data of a collection from MongoDb into a dataframe

cursor = Player_Details.find()
df_Player_Details =  pd.DataFrame(list(cursor))
df_Player_Details.head()


# In[74]:


#Checking the shape of the dataframe
df_Player_Details.shape


# In[75]:


##Deleting _id column (created in MongoDb as a unique identification for documents of a collection) as it is not required
del df_Player_Details['_id']

df_Player_Details.shape


# In[76]:


##Checking for null values
pd.isnull(df_Player_Details).sum()


# In[77]:


# removing column Nation Jersey Number as it has a large number of null values and is not so important for achieving the objective.

del df_Player_Details['nation_jersey_number']

df_Player_Details.shape


# In[78]:


#Similarly attributes like player_url, body_type, real_face, player_tags, loaned_from, joined, contract_valid_until can be deleted

df_Player_Details = df_Player_Details.drop(['player_url','body_type','real_face','player_tags','loaned_from', 'joined','team_jersey_number','contract_valid_until'], axis = 1)

df_Player_Details.shape


# In[79]:


#Again checking for null values
pd.isnull(df_Player_Details).sum()


# In[82]:


#Replacing null values in 'release_clause_eur' column with mean value of that column.
mean_value = df_Player_Details['release_clause_eur'].mean()

df_Player_Details=df_Player_Details.fillna(mean_value)
df_Player_Details.shape


# In[83]:


#All null records have been handled
pd.isnull(df_Player_Details).sum()


# ##### End of code by Pooja Rakate for Data pre-processing of Player Details data

# ##### Start of code by Mayuresh Londhe for Data pre-processing for Player Positions data

# In[10]:


#Checking whether we are able to find records from mongo db 
for record in PPRANK.find({'Player_id':158023}):
    print(record)


# In[11]:


#Finding out Number of Goalkeepers 
# As a Goalkeeper plays only in one position dataset has Null values for all other postions for GoalKeeper.
#Due to which 2036 is the number of goalkeepers in dataset and in all Postion Columns there will 2036 Null values. 
#Which we will be replacing by 0

PPRANK.count_documents({'ls':None})  # counting null value in left Striker (LS) postion column


# In[12]:


###reading data of a collection from MongoDb into a dataframe
cursor1 = PPRANK.find()
df_pos_rank =  pd.DataFrame(list(cursor1))
print(df_pos_rank)


# In[42]:


#Count of Null Values from All the columns

pd.isnull(df_pos_rank).sum()


# In[13]:


# Removed Object Id

del df_pos_rank["_id"]
df_pos_rank.head()


# In[14]:


# # filling NA values: only goalkeeprs have missing data for these variables as these are ratings for positions other than goalkeeper.
#But we cannot delete rows having NA values as the goalkeepers have data for other variables. Hence, replacing them with 0.
df_pos_rank=df_pos_rank.fillna(0)
df_pos_rank.head()


# In[15]:


# checking whether fillna is successful.
#it is successful
print(df_pos_rank)


# ##### End of code by Mayuresh Londhe for Data pre-processing of Player positions data

# ##### Start of code by Ankit Rungta for Data pre-processing of AM attributes data

# In[16]:


####Checking whether we are able to find records from mongo db 

pprint.pprint(AM_Attributes.find({"item": {"Player_id":20801}}))


# In[21]:


##reading data of a collection from MongoDb into a dataframe
cursor2 = AM_Attributes.find()
df_AM_Attr =  pd.DataFrame(list(cursor2))
print(df_AM_Attr)


# In[22]:


####Removing unnecessary column id that was added while loading data from MongoDb into dataframe
df_AM_Attr = df_AM_Attr.drop(columns="_id")
df_AM_Attr.head()


# In[23]:


#Checking for null Values 
pd.isnull(df_AM_Attr)


# In[24]:


pd.isnull(df_AM_Attr).sum()


# In[25]:


#Having null values in Pace,Shooting, and dribbling 
##Replacing these NA values with 0 as the data is for attackers and mid-fielders and the null values are for Goal-keepers and Defender

df_AM_Attr=df_AM_Attr.fillna(0)
df_AM_Attr.head()


# In[26]:


####Checking for null values

pd.isnull(df_AM_Attr).sum()


# ##### End of code by Ankit Rungta for Data pre-processing of AM attributes data

# ##### Start of code by Priyanka Chimthankar for Data pre-processing of DG attributes data

# In[27]:


###Checking whether we are able to find records from mongo db 
cursor3 = DG_Attributes.find()
df_DG_Attr =  pd.DataFrame(list(cursor3))
print(df_DG_Attr)


# In[28]:


####Removing id column as it is not needed
df_DG_Attr=df_DG_Attr.drop(columns= "_id")
df_DG_Attr


# In[29]:


####Checking for null values in the collection
pd.isnull(df_DG_Attr)


# In[30]:


####Count of null values in all columns
pd.isnull(df_DG_Attr).sum()


# In[31]:


#### Replacing NA values with 0 as the data consists of all the Player's data and have null values where the player is not eligible for that attribute
## gk_speed is null for the players except for goalkeepers
##defending and physic is null for the goalkeepers,hence replacing with 0
df_DG_Attr=df_DG_Attr.fillna(0)
df_DG_Attr


# In[32]:


pd.isnull(df_DG_Attr).sum()


# ##### End of code by Priyanka Chimthankar for Data pre-processing of DG attributes data

# # Part 2 - Reading and Writing Processed Data to PostgreSql 

# In[ ]:


#Connect to PostgreSQL and create a database


# In[102]:


try:
    #Creating a connection variable to connect to PostgreSQL
    connection = pg.connect(database="postgres", user='postgres', password='password', host='127.0.0.1', port= '5432')
    
    connection.autocommit = True
    
    cursor = connection.cursor()
    
    ##Creating Database
    cursor.execute("create database PES_Players_2020")
    
    cursor.execute("SELECT version();")
    # Fetch result
    record = cursor.fetchone()
    print("You are connected to - ", record, "\n")
    print("Database created successfully !!")
    
except:
    print("Error while connecting to PostgreSQL !!")


# ##### Start of code by Pooja Rakate for Writing  and reading Player Details data to and from PostgreSQL

# In[36]:


##Extablishing SQLAlchemy engine connection to create a table in the database

engine = sqla.create_engine('postgresql://postgres:password@localhost:5432/pes_players_2020')
con_engine = engine.connect()


# In[148]:


##Creating a table Player_Details in database 'pes_players_2020' created in above step
#If table already exists the it will raise an error. we can change this to append if table already exists 
#by changing the value to if_exists argument

df_Player_Details.to_sql('Player_Details', con_engine, if_exists = 'fail', index = False)

#Check if the table is created
print(engine.table_names())


# In[37]:


##Fetching top 1000 rows from Player_Details table from postgreSql and displaying it here

result = engine.execute('select * from "Player_Details" LIMIT 1000').fetchall()
num_records = 0

for r in result:
    num_records= num_records+1
    print(r)
    
print(num_records)


# In[62]:


##Storing Tables from PostgreSQL into dataframe in Pandas for further visualisation purposes

df_Player_Det_PG = pd.read_sql_table("Player_Details", con_engine);

df_Player_Det_PG.shape


# ##### End of code by Pooja Rakate for Writing  and reading Player Details data to and from PostgreSQL

# ##### Start of code by Mayuresh Londhe for Writing  and reading Player Positions data to and from PostgreSQL

# In[155]:


#Creating Player_Position table

df_pos_rank.to_sql('Player_Position', con_engine, if_exists = 'fail', index = False)

#Check if the table is created
print(engine.table_names())


# In[38]:


##Fetching top 1000 rows from Player_Position table from postgreSql and displaying it here

result1 = engine.execute('select * from "Player_Position" LIMIT 1000').fetchall()
num_records1 = 0

for r in result1:
    num_records1 = num_records1 + 1
    print(r)
    
print(num_records1)


# In[63]:


##Storing Tables from PostgreSQL into dataframe in Pandas for visualisation purposes

df_Player_Pos_PG = pd.read_sql_table("Player_Position", con_engine);

df_Player_Pos_PG.shape


# ##### End of code by Mayuresh Londhe for Writing  and reading Player Positions data to and from PostgreSQL

# ##### Start of code by Ankit Rungta for Writing  and reading AM Player Attributes data to and from PostgreSQL

# In[156]:


#Creating AM_Player_Attributes table

df_AM_Attr.to_sql('AM_Player_Attributes', con_engine, if_exists = 'fail', index = False)

#Check if the table is created
print(engine.table_names())


# In[39]:


##Fetching top 1000 rows from AM_Player_Attributes table from postgreSql and displaying it here

result2 = engine.execute('select * from "AM_Player_Attributes" LIMIT 1000').fetchall()
num_records2 = 0

for r in result2:
    num_records2 = num_records2 + 1
    print(r)
    
print(num_records2)


# In[61]:


##Storing Tables from PostgreSQL into dataframe in Pandas for visualisation purposes

df_AM_Attr_PG = pd.read_sql_table("AM_Player_Attributes", con_engine);

df_AM_Attr_PG.shape


# ##### End of code by Ankit Rungta for Writing  and reading AM Player Attributes data to and from PostgreSQL

# ##### Start of code by Priyanka Chimthankar for Writing  and reading DG Player Attributes data to and from PostgreSQL

# In[157]:


#Creating DG_Player_Attributes table

df_DG_Attr.to_sql('DG_Player_Attributes', con_engine, if_exists = 'fail', index = False)

#Check if the table is created
print(engine.table_names())


# In[40]:


##Fetching top 1000 rows from DG_Player_Attributes table from postgreSql and displaying it here


result3 = engine.execute('select * from "DG_Player_Attributes" LIMIT 1000').fetchall()
num_records3 = 0

for r in result3:
    num_records3 = num_records3 + 1
    print(r)
    
print(num_records3)


# In[64]:


##Storing Tables from PostgreSQL into dataframe in Pandas for visualisation purposes

df_DG_Attr_PG = pd.read_sql_table("DG_Player_Attributes", con_engine);

df_DG_Attr_PG.shape


# ##### End of code by Priyanka Chimthankar for Writing  and reading DG Player Attributes data to and from PostgreSQL

# 

# # Part 3 - Creating a single Dataframe of all the four tables

# In[48]:


###concatenating all the dataframes df_Player_Det_PG, df_Player_Pos_PG, df_AM_Attr_PG and df_DG_Attr_PG 
#in one DataFrame by joining on the Player_id column to use it for visualisation

df_list=[df_Player_Det_PG, df_Player_Pos_PG, df_AM_Attr_PG, df_DG_Attr_PG]

df_PES_Visualisation = rd(lambda left,right: pd.merge(left,right,on='Player_id'), df_list)


# In[135]:


df_PES_Visualisation.head()


# In[136]:


df_PES_Visualisation.shape


# In[175]:


##Converting the final data frame into csv file to share it with team members so that each member 
#can work on visualisations of their respective parts.

df_PES_Visualisation.to_csv('D:/Pooja_NCI/1_Semester/DataBase and Analytical programming/Project/Code from everyone/PES_Final_for_Visualisation.csv')


# 

# # Part 4 - Visualisation

# In[8]:


# Summary function
df_PES_Visualisation.describe()


# In[ ]:





# In[9]:


##Age wise analysis
fig, ax = plt.subplots(1,2,figsize=(18,6))

sns.distplot(ax=ax[0], a=df_PES_Visualisation.age, kde=False,color="k")
sns.scatterplot(y='overall', x='age', data=df_PES_Visualisation, ax=ax[1], color="g")

ax[0].set_title("Age Distribution in PES 2020")
ax[1].set_title("Overall Ranking distribution for Player's age")


for i in range(2):
    ax[i].set_ylabel("Number")


# In[7]:


#Univariate distribution of Observations using Distplot & Fitness of player
fig, ax = plt.subplots(1,3,figsize=(18,6))

sns.distplot(ax=ax[0], a=df_PES_Visualisation.height_cm, kde=False,color="k")
sns.distplot(ax=ax[1], a=df_PES_Visualisation.weight_kg, kde=False,color="k")
sns.regplot(y="height_cm", x="weight_kg", data=df_PES_Visualisation, ax=ax[2],color="b")

ax[0].set_title("Height in CM Distribution in PES 2020")
ax[1].set_title("Weight in kg Distribution in PES 2020")
ax[2].set_title("Height and Weight Comparison for different players")


for i in range(3):
    ax[i].set_ylabel("Number")


# In[10]:


#PLotting Players with Stamina,Strength and Physic greater than 85

#Defining function
def scatter3D(x , y , z , name_of_the_player , xlabel , ylabel , zlabel , plot_title):
    zoom_camera = dict(up=dict(x=0, y=0, z=1),center=dict(x=0, y=0, z=0),eye=dict(x=2, y=2, z=0.1))
    
    plot = go.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode = 'markers',
        text  = name_of_the_player,
        marker = dict(
            size = 12,
            color = z,
            colorscale = 'sunset',
            showscale = True,
            line = dict(
                color = 'rgba(225 , 225 , 225 , 0.14)',
                width = 0.5
            ),
            opacity = 0.8
        )
    )
    
    layout = go.Layout(
        title = plot_title,
        scene = dict(
            camera = zoom_camera,
            xaxis = dict(title  = xlabel),
            yaxis = dict(title  = ylabel),
            zaxis = dict(title  = zlabel)
        )
    )
    data = [plot]
    fig = go.Figure(data = data , layout = layout)
    py.offline.iplot(fig)

#Function Call
scatter3D(df_PES_Visualisation['stamina'].where(df_PES_Visualisation['stamina'] > 85),
          df_PES_Visualisation['strength'].where(df_PES_Visualisation['strength'] > 85),
          df_PES_Visualisation['physic'].where(df_PES_Visualisation['physic'] > 85),
          df_PES_Visualisation['short_name'],
         'Stamina' , 
         'Strength',
         'Physic',
         'Players with Stamina,Strength and Physic greater than 85')  


# In[11]:


#Wage and Value of Player

sns.regplot(y='value_eur', x='wage_eur', data=df_PES_Visualisation,color="b", label = 'Wage and Value Comparison for different players')


# In[12]:


#Analysis of players Wage by their work_rate  and playing position
fig, ax = plt.subplots(2,1,figsize=(18,18))
ax = ax.ravel()

##Remove
sns.barplot(data=df_PES_Visualisation, y="wage_eur", x="team_position", ax=ax[0])
##

sns.barplot(data=df_PES_Visualisation, y="work_rate", x="wage_eur", ax=ax[1])

ax[0].set_title("Wage by Player's Position")
ax[1].set_title("Wage by Work rate")


fig.tight_layout()


# In[13]:


#Different Attributes of Player in Relation with Overall

fig, ax = plt.subplots(2,3,figsize=(18,12))
ax = ax.ravel()

sns.scatterplot(y="overall", x="potential", data=df_PES_Visualisation, color="brown", ax=ax[0])
sns.scatterplot(y='overall', x='physic', data=df_PES_Visualisation, color="r", ax=ax[1])
sns.scatterplot(y="overall", x="shooting", data=df_PES_Visualisation, color="black", ax=ax[2])
sns.scatterplot(y='overall', x='passing', data=df_PES_Visualisation, color="g", ax=ax[3])
sns.scatterplot(y="overall", x="dribbling", data=df_PES_Visualisation, color="b", ax=ax[4])
sns.scatterplot(y='overall', x='defending', data=df_PES_Visualisation, color="y", ax=ax[5])


ax[0].set_title("Potential vs Overall")
ax[1].set_title("Physic vs Overall")
ax[2].set_title("Shooting vs Overall")
ax[3].set_title("Passing vs Overall")
ax[4].set_title("Dribbling vs Overall")
ax[5].set_title("Defending vs Overall")

fig.suptitle("Different Attributes of Player in Relation with Overall", size=20)


# In[13]:


#Distribution of overall rating

plt.figure(figsize=(8,8))
sns.distplot( a=df_PES_Visualisation.overall, kde=False)
plt.title("Distribution of Overall Rating")
plt.ylabel("frequency")


# In[14]:


##Comparison of Defending rating for different player positions

def plot(x  , y  , data , rows , cols):
        color_used = []
        n = 0
        for feature in y:
            
            for i in range(1000):
                colour = 'lightblue'
                if colour not in color_used:
                    color_used.append(colour)
                    break
    
            n += 1 
            plt.subplot(rows , cols , n)
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
            sns.regplot(x  = x , y = feature , data = data ,color = colour)
            
vals = ['st' , 'cam' , 'cb']
plt.figure(1 , figsize = (15 , 6))
plot(x = 'defending' , y = vals , data = df_PES_Visualisation , 
                         rows = 1 , cols = 3)
plt.show()


# In[15]:


##Top 10 Nations

labels = ['England', 'Germany', 'Spain', 'France', 'Argentina','Brazil','Italy','Columbia','Japan','Netherlands']

sizes = df_PES_Visualisation['nationality'].value_counts().head(10)
colors = plt.cm.copper(np.linspace(0, 1, 5))
plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(sizes, colors = colors, shadow = True,labels=labels)
plt.title('Top 10 Nations consisting maximum number of Total PES Players', fontsize = 20)
plt.legend()
plt.show()


# In[16]:


#Distribution of wage for top 10 players
plt.rcParams['figure.figsize'] = (15, 10)
df_best_players=df_PES_Visualisation.sort_values(by="overall",ascending=False).head(10)
sns.barplot(data=df_best_players,y="wage_eur",x="short_name" ,color = 'lightblue')
plt.xlabel('Name of the Players', fontsize = 12)
plt.ylabel('Wage Range of Players', fontsize = 12)
plt.title('Wage Distribution of top 10 Rated Players', fontsize = 15)
plt.show()


# In[17]:


#Percentage of players in different attacking role

attacker = ['RW', 'LW', 'ST', 'CF', 'LS', 'RS', 'RF', 'LF'] # List of attacker postions

sample = df_PES_Visualisation.query('team_position in @attacker')   #query to select attackers from team postion column as specified in team postion list 

#Creating pie type object using pie() method
figure_Attacker = px.pie(sample, names='team_position',  # Using pie method of plotly to visualize
             color_discrete_sequence=px.colors.sequential.Plasma_r, #specifying color gradient sequence for pie
             title='Percentage of players in Attacker Role')  # Title for the plot

figure_Attacker.update_traces(textposition='inside', textinfo='percent+label') # specifying textpostion and type of text i.e. % and a lable

#specifying Layout of the plot using paper_bgcolor plot_bgcolor font
figure_Attacker.update_layout(paper_bgcolor='rgba(0,0,0,0)',  #specifying backgroud color for paper
                  plot_bgcolor='rgba(0,0,0,0)',  #specifying backgroud color for plot
                  font=dict(family='Cambria, monospace', size=12, color='darkorange'))  #specifying font size and color of the text

figure_Attacker.show()


# In[18]:


##Percentage of players in different Midfielder roles

midfielder = ['CAM', 'RCM', 'CDM', 'LDM', 'RM', 'LCM', 'LM', 'RDM', 'RAM','CM', 'LAM']    # List of Midfielder postions

sample = df_PES_Visualisation.query('team_position in @midfielder')    #query to select Midfielders from team postion column as specified in team postion list 

#Creating pie type object using pie() method
figure_midfielder = px.pie(sample, names='team_position',   # Using pie method of plotly to visualize
             color_discrete_sequence=px.colors.sequential.Viridis_r, #specifying color gradient sequence for pie
             title='Percentage of players in Different Midfielder Role')  # Title for the plot
figure_midfielder.update_traces(textposition='inside', textinfo='percent+label')  # specifying textpostion and type of text i.e. % and a lable

#specifying Layout of the plot using paper_bgcolor plot_bgcolor font
figure_midfielder.update_layout(paper_bgcolor='rgba(0,0,0,0)', #specifying backgroud color for paper
                  plot_bgcolor='rgba(0,0,0,0)',      #specifying backgroud color for plot
                  font=dict(family='Cambria, monospace', size=14, color='blue')) #specifying font size and color of the text
figure_midfielder.show()


# In[19]:


# ##Percentage of players in different Defender roles

defender = ['LCB', 'RCB', 'LB', 'RB', 'CB', 'RWB', 'LWB'] # List of defender postions

query_plot = df_PES_Visualisation.query('team_position in @defender')  #query to select defedenders from team postion column as specified in team postion list

#Creating pie type object using pie() method
figure_defender = px.pie(query_plot, names='team_position', # Using pie method of plotly to visualize
             color_discrete_sequence=px.colors.sequential.Magma_r, #specifying color gradient sequence for pie 
             title='Percentage of Players in Different Defender Role')     # Title for the plot

figure_defender.update_traces(textposition='inside', textinfo='percent+label') # specifying textpostion and type of text i.e. % and a lable

#specifying Layout of the plot using paper_bgcolor plot_bgcolor font
figure_defender.update_layout(        
                  paper_bgcolor='rgba(0,0,0,0)',#specifying backgroud color for paper
                  plot_bgcolor='rgba(0,0,0,0)', #specifying backgroud color for plot 
                  font=dict(family='Cambria, monospace', size=14, color='firebrick') #specifying font size and color of the text
                  )
figure_defender.show()


# In[20]:


#Selecting the best team from the user defined player positions

def best_formation_selection(position):  # creted function definition 
    df_PES_Visualisation_copy = df_PES_Visualisation.copy()
    team = []
    
    for i in position:   # for loop to iterate over positions specified in function call 
        
        # computing the max player count for player position and appending it to a team
        team.append([i,df_PES_Visualisation_copy.loc[[df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].idxmax()]]['short_name'].to_string(index = False), df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].max()])
        
        # Dropping the players appended to team so that they will be not be selected again in team
        df_PES_Visualisation_copy.drop(df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].idxmax(), inplace = True)
    
        #returning team by reshaping it to array of 11*3  
    return pd.DataFrame(np.array(team).reshape(11,3), columns = ['Position', 'Player', 'Overall'])
    

# 4-3-3         4 Defenders  3 midfielders 3 forwrds 1 goalkeeper
formation_433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']
print ('Formation: 4-3-3')
#Function Call
print (best_formation_selection(formation_433))


# In[21]:


# 3-5-2   2 forwards 5 midfielders 3 defenders 1 Goalkeeper
formation_352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']
print ('3-5-2')
print (best_formation_selection(formation_352))


# In[22]:


##Player's international reputation
df_PES_Visualisation[df_PES_Visualisation['international_reputation'] == 5][['short_name', 'age', 'club', 'nationality','overall']].style.background_gradient('bwr')


# In[ ]:





# In[23]:


##Clubwise Best team formation

def club_best_formation_selection(position,club_name):  # created function definition 
    df_PES_Visualisation_copy = df_PES_Visualisation.copy()
    team = []
    
    df_PES_Visualisation_copy = df_PES_Visualisation_copy.loc[df_PES_Visualisation_copy['club'] == club_name]                                                 
    
    for i in position:   # for loop to iterate over positions specified in function call 
        
        # computing the max player count for player position and appending it to a team
        team.append([i,df_PES_Visualisation_copy.loc[[df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].idxmax()]]['short_name'].to_string(index = False), 
                     df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].max()])
        
        # Dropping the players appended to team so that they will be not be selected again in team
        df_PES_Visualisation_copy.drop(df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].idxmax(), inplace = True)
    
        #returning team by reshaping it to array of 11*3
    print("Team Selection Based on Club : ",club_name)
    return pd.DataFrame(np.array(team).reshape(11,3), columns = ['Position', 'Player', 'Overall'])
    

# 4-3-3         4 Defenders  3 midfielders 3 forwrds 1 goalkeeper
formation_433 = ['GK', 'LB', 'RB', 'LCB', 'RCB', 'LCM', 'CDM', 'RCM', 'LW', 'CF', 'RW']
print ('4-3-3')
#Function Call

print (club_best_formation_selection(formation_433,'Real Madrid'))


# In[ ]:





# In[24]:


##Nationwise best team selection

def nation_best_formation_selection(position,nation_name):  # created function definition 
    df_PES_Visualisation_copy = df_PES_Visualisation.copy()
    team = []
    
    df_PES_Visualisation_copy = df_PES_Visualisation_copy.loc[df_PES_Visualisation_copy['nationality'] == nation_name]                                                 
    
    for i in position:   # for loop to iterate over positions specified in function call 
        
        # computing the max player count for player position and appending it to a team
        team.append([i,df_PES_Visualisation_copy.loc[[df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].idxmax()]]['short_name'].to_string(index = False), df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].max()])
        
        # Dropping the players appended to team so that they will be not be selected again in team
        df_PES_Visualisation_copy.drop(df_PES_Visualisation_copy[df_PES_Visualisation_copy['team_position'] == i]['overall'].idxmax(), inplace = True)
    
        #returning team by reshaping it to array of 11*3 
    print("Team Selection Based on Nation : ",nation_name)
    return pd.DataFrame(np.array(team).reshape(11,3), columns = ['Position', 'Player', 'Overall'])
    

# 4-3-3         4 Defenders  3 midfielders 3 forwrds 1 goalkeeper
formation_433 = ['GK', 'LB', 'RB', 'LCB', 'RCB', 'LCM', 'CDM', 'RCM', 'LW', 'ST', 'RW']
print ('4-3-3')
#Function Call

print (nation_best_formation_selection(formation_433,'Spain'))


# In[ ]:




