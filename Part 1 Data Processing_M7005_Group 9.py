import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir("E:/HKU/MSDA7005")

###############################################################################
#1. Process the data of depression
#Read the file
cognition_raw_data = pd.read_stata("2018全国追踪调查/CHARLS2018r/Cognition.dta")
#We need the Depression Scale, which includes questions DC009 - DC018
cog_columns = ['ID','householdID','communityID','dc009','dc010','dc011','dc012','dc013','dc014','dc015','dc016','dc017','dc018']
cog_selected = cognition_raw_data[cog_columns]
#We first transfer all the str results into int type.
question_columns = ['dc009','dc010','dc011','dc012','dc013','dc014','dc015','dc016','dc017','dc018']
for columns in question_columns:
    cog_selected.loc[:,columns] = cog_selected[columns].apply(lambda x: int(str(x)[0]) if pd.notnull(x) and str(x)[0].isdigit() else np.nan)
#For the answer values of 1-4, a higher number indicates a deeper degree. Among the 10 questions, except for two questions, dc0016 and dc0013, which are positive, the other 8 are negative. Therefore, we need to assign scores to each of these questions. If it is a positive question, then 1-4 are assigned a value of 3-0 respectively, indicating a decreasing level of depression. Whereas, in the negative questions, 1-4 are assigned a value of 0-3 respectively, indicating an increasing level of depression.
positive_question = ['dc013','dc016']
negative_question = ['dc009','dc010','dc011','dc012','dc014','dc015','dc017','dc018']
for columns in question_columns:
    if columns in positive_question:
        score_map = {1: 3, 2: 2, 3: 1, 4: 0, 8: np.nan, 9: np.nan}
        cog_selected.loc[:,columns] = cog_selected[columns].apply(lambda x: score_map.get(x))
    else:
        score_map = {1: 0, 2: 1, 3: 2, 4: 3, 8: np.nan, 9: np.nan}
        cog_selected.loc[:,columns] = cog_selected[columns].apply(lambda x: score_map.get(x))
#We then calculate the mean of depression score. The higher the score is, the more depression the interviewee has.
depression_df = cog_selected.iloc[:,:3]
depression_df['Depression'] = cog_selected[question_columns].mean(axis=1, skipna=True)
depression_df = depression_df.dropna(subset = ['Depression'])[['ID','Depression']]
depression_df['ID'] = depression_df['ID'].apply(lambda x: int(str(x).lstrip('0')) if str(x).startswith('0') else int(x))


###############################################################################
#2. Process the data of health
#Read the file
health_raw_data = pd.read_stata("2018全国追踪调查/CHARLS2018r/Health_Status_and_Functioning.dta")
#We need the health self-evaluation data, which is question DA002
health_columns = ['ID','householdID','communityID','da002']
health_selected = health_raw_data[health_columns]
health_selected.loc[:,'da002'] = health_selected.loc[:,'da002'].apply(lambda x: int(str(x)[0]) if pd.notnull(x) and str(x)[0].isdigit() else np.nan)
health_df = health_selected.dropna(subset = ['da002']).rename(columns={'da002': 'Health'})[['ID','Health']]
health_df['ID'] = health_df['ID'].apply(lambda x: int(str(x).lstrip('0')) if str(x).startswith('0') else int(x))


###############################################################################
#3. Process the data of work
#Read the file
work_raw_data = pd.read_stata("2018全国追踪调查/CHARLS2018r/Work_Retirement.dta")
#We need to know whether the interviewee has work to do currently, which includes question FC008, FC001, FA002_W4. If the answer to any of the 3 questions is 1 yes, then we can say that the interviewee has work to do now.
work_columns = ['ID','householdID','communityID','fc008','fc001','fa002_w4']
work_selected = work_raw_data[work_columns]
work_selected['Work'] = work_selected[['fc008', 'fc001', 'fa002_w4']].apply(lambda x: 1 if("1 Yes" in x.values) else 0,axis = 1)
work_df = work_selected[['ID','Work']]
work_df['ID'] = work_df['ID'].apply(lambda x: int(str(x).lstrip('0')) if str(x).startswith('0') else int(x))


###############################################################################
#4. Process the data of education, marriage, sex , area and religion
personal_df = pd.read_stata("2018全国追踪调查/CHARLS2018r/Demographic_Background.dta")
personal_df = personal_df[['ID','householdID','communityID', 'bd001_w2_4', 'be001', 'bg002_w4','ba000_w2_3', 'bb000_w3_2']].reset_index(drop = True).dropna()

personal_df['bd001_w2_4'] = personal_df['bd001_w2_4'].map({
    "1 No Formal Education (Illiterate)": 1,
    "2 Did not Finish Primary School": 2,
    "3 Sishu/Home School": 3,
    "4 Elementary School": 4,
    "5 Middle School": 5,
    "6 High School": 6,
    "7 Vocational School": 7,
    "8 Two-/Three-Year College/Associate Degree": 8,
    "9 Four-Year College/Bachelor’s Degree": 9,
    "10 Master’s Degree": 10,
    "11 Doctoral Degree/Ph.D.": 11
}).astype('category')

personal_df['be001'] = personal_df['be001'].map({
    "1 Married with Spouse Present": 1,
    "2 Married But Not Living with Spouse Temporarily for Reasons Such as Work": 1,
    "3 Separated": 0,
    "4 Divorced": 0,
    "5 Widowed": 0,
    "6 Never Married": 0
}).astype('category')

personal_df['bg002_w4'] = personal_df['bg002_w4'].map({
    "1 Yes": "1",
    "2 No": "0"
}).astype('category')

personal_df['ba000_w2_3'] = personal_df['ba000_w2_3'].map({
    "1 Male": "0",
    "2 Female": "1"
}).astype('category')

personal_df = personal_df.rename(columns = {"bd001_w2_4":"Level of Education", "be001": "Marriage", "bg002_w4": "Religious Belief",'ba000_w2_3':'Gender'})[['ID','Level of Education','Marriage','Religious Belief','Gender']]
personal_df['ID'] = personal_df['ID'].apply(lambda x: int(str(x).lstrip('0')) if str(x).startswith('0') else int(x))


###############################################################################
#4. Process the data of household income and area
income_df = pd.read_csv("2018全国追踪调查/output.csv")
income_df = income_df.rename(columns = {"hh4itot":'Family Income'})

general_df = pd.read_excel("2018全国追踪调查/CHARLS.xlsx", sheet_name= 'CHARLS')
rural_df = general_df[['ID','rural']].drop_duplicates(subset=['ID'])
rural_df = rural_df.rename(columns = {"rural":'Rural'})


###############################################################################
#5.Merge all the data
merged_df_1 = pd.merge(income_df, rural_df, on = 'ID', how = 'inner')
merged_df_2 = pd.merge(merged_df_1, work_df, on = 'ID', how = 'inner')
merged_df_3 = pd.merge(merged_df_2, depression_df, on = 'ID', how = 'inner')
merged_df_4 = pd.merge(merged_df_3, health_df, on = 'ID', how = 'inner')
merged_df_5 = pd.merge(merged_df_4, personal_df, on = 'ID', how = 'inner')

merged_df_5.to_excel('Combined Data.xlsx')







