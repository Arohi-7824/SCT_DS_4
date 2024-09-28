import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\arohi\OneDrive\Desktop\Task_4\US_Accidents_Dec21_updated.csv")
print(df.head())
print("The number od columns are",df.columns)
print(df.describe)
print(df.info())

numerics = ['int16','int32','int64','float32','float64']
numeric_df = df.select_dtypes(include = numerics)
print(f"There are {len(numeric_df.columns)} number of numeric columns in the DataFrame.")

df.isna()

df1=df[df['State']=='CA']
print(df1.head())

df1['Weather_Condition'].value_counts()


d1f=df1.dropna(subset=['Precipitation(in)'])    
df1=df1.dropna(subset=['Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Direction', 'Wind_Speed(mph)',
                     'Weather_Condition'])
df1.isna().sum()/len(df1)*100

df_num=df1.select_dtypes(np.number)
col_name=[]
length=[]

for i in df_num.columns:
    col_name.append(i)
    length.append(len(df_num[i].unique()))
df_2=pd.DataFrame(zip(col_name,length),columns=['feature','count_of_unique_values'])
df_2


# Creating the heatmap
corr_matrix = df_num.corr()
fig, ax = plt.subplots(figsize=(15, 9))
cax = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
fig.colorbar(cax)

ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix.columns)):
       text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha='center', va='center', color='black')

plt.tight_layout()
plt.show()


accidents_by_cities = df1['City'].value_counts()
accidents_by_cities
accidents_by_cities[:10]

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('Set2', n_colors=10)
accidents_by_cities[:10].plot(kind='bar', ax=ax, color=colors)
ax.set_title('Top 10 Cities by Number of Accidents', fontsize=16, pad=15)
ax.set_xlabel('Cities', fontsize=12)
ax.set_ylabel('Accidents Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Create the figure and axis with equal aspect ratio
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))
labels = ['Minor', 'Moderate', 'Severe', 'Critical']
# Define the severity counts (as an example)
accidents_severity = [30, 45, 15, 10]
plt.pie(accidents_severity, labels=labels,
        autopct='%1.1f%%', pctdistance=0.85, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
circle = plt.Circle((0, 0), 0.5, color='white')
plt.gca().add_artist(circle)
ax.set_title("Accidents by Severity", fontdict={'fontsize': 16})
plt.tight_layout()
plt.show()

df1['Start_Time'] = pd.to_datetime(df1['Start_Time'], errors='coerce')
df1 = df1.dropna(subset=['Start_Time'])
accidents_by_hour = df1['Start_Time'].dt.hour.value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=accidents_by_hour.index, y=accidents_by_hour.values, color='blue')  # Use one color
plt.title('Accidents Count By Hour of the Day', fontsize=16, fontweight='bold')
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.show()


df1['Start_Time'] = pd.to_datetime(df1['Start_Time'], errors='coerce')
df1['End_Time'] = pd.to_datetime(df1['End_Time'], errors='coerce')
df1 = df1.dropna(subset=['Start_Time'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df1['End_Time'].dt.hour, bins=24, color='teal', edgecolor='black')
plt.xlabel("Hour of End Time", fontsize=14)
plt.ylabel("Number of Occurrences", fontsize=14)
plt.title('Accidents Count By End Time of Day', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# Plot the top 20 weather conditions with accidents
Weather_Condition = df1['Weather_Condition'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))

# Use kind='barh' for horizontal bars
Weather_Condition.sort_values(ascending=False)[:20].plot(kind='barh', color='coral', edgecolor='black', ax=ax)

# Set the title and axis labels
ax.set_title('Top 20 Weather Conditions at Time of Accident Occurrence', fontsize=16, fontweight='bold')
ax.set_xlabel('Accidents Count', fontsize=14)
ax.set_ylabel('Weather Conditions', fontsize=14)

# Rotate y-tick labels to be horizontal
plt.yticks(rotation=0)

# Add gridlines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


#Check if the necessary columns exist
if 'Start_Lat' in df.columns and 'Start_Lng' in df.columns:
    
    # Create a hexbin plot to visualize the density of accidents
    plt.figure(figsize=(12, 8))
    plt.hexbin(df['Start_Lng'], df['Start_Lat'], gridsize=50, cmap='coolwarm', mincnt=1)

    # Add color bar to indicate density
    plt.colorbar(label='Number of Accidents')

    plt.title('Heatmap of Traffic Accident Density')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    
    # Show plot
    plt.show()
else:
    print("Make sure your dataset contains 'Start_Lat' and 'Start_Lng' columns.")
    
plt.figure(figsize=(10, 6))
sns.violinplot(x='Severity', y='Wind_Speed(mph)', data=df, palette='coolwarm')
plt.title('Violin Plot of Wind Speed by Accident Severity')
plt.xlabel('Severity')
plt.ylabel('Wind Speed (mph)')
plt.grid(axis='y')
plt.show()

# Swarm plot for humidity by accident severity
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Severity', y='Humidity(%)', data=df, hue='Severity', palette='coolwarm', dodge=True)
plt.title('Swarm Plot of Humidity by Accident Severity')
plt.xlabel('Severity')
plt.ylabel('Humidity (%)')
plt.grid(axis='y')
plt.show()

