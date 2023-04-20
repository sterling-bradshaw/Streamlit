import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.express as px
import re

st.write("""
## Bayes Rule Application: 
### P(Year of Birth | Name and Sex)
""")

usa = pd.read_csv('names_usa.csv')

# CSS to inject contained in a string
hide_table_row_index = """
             <style>
             thead tr th {display:none}
             tbody th {display:none}
             </style>
             """

# # Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)



prior_type = st.radio(
        "Choose a prior distribution for year of birth",
        ("Emperical", "Uniform", "Normal Low Certainty", "Normal High Certainty"))

a = usa.groupby(['year']).sum()
years = np.arange(1910,2022)
a = a.iloc[:,0]/a.iloc[:,0].sum()
emp_prior = pd.Series(data=a, index=years)
un_prior = pd.Series(data=np.repeat(1/112,112),index=years)
rv = norm(loc=1980, scale=50)
nl_prior = pd.Series(data = rv.pdf(years)/(rv.pdf(years).sum()), index=years)
rv = norm(loc=1980, scale=20)
nh_prior = pd.Series(data = rv.pdf(years)/(rv.pdf(years).sum()), index=years)

# prior_type = st.radio(
#         "Choose a prior distribution for year of birth",
#         ("Emperical", "Uniform", "Normal Low Confidence", "Normal High Confidence"))


with st.expander('See comparison of prior'):
    fig1 = plt.figure(figsize=(9,6))
    plt.plot(years, emp_prior, label='Emperical')
    plt.plot(years, un_prior, label='Uniform')
    plt.plot(years, nl_prior, label='Normal Low Certainty')
    plt.plot(years, nh_prior, label='Normal High Certainty')
    plt.xlabel('Years')
    plt.ylabel('Probability')
    plt.title('Prior probability of birth year')
    plt.ylim((0,.023))
    plt.xlim((1910,2021))
    plt.legend(loc=(1.01,0))
    st.pyplot(fig1)

pattern = st.text_input("Type a name to search", placeholder="Enter a name or regular expression")

regex = st.radio(
    "Is your search a regular expression?",
    ('No','Yes'))
    
cond_sex = st.radio("Do you want to condition on a sex?",
   ("No", "Yes"))
    
if cond_sex == "Yes":
        sex = st.radio("Which sex?",
            ("M", "F"))
        sex = set(sex)
    
else:
        sex = set(["M", "F"])

if len(pattern) > 0:
    df = usa[usa.sex.isin(sex)]
    df = df.merge(df.groupby('year').sum().reset_index(), on='year')
    df['pname'] = df['n_x']/df['n_y']
    unique_names = set(df['name'])
    if regex=='Yes':
        keep = {name for name in unique_names if re.match(pattern, name)}
    else:
        keep = {pattern}
    nameDF = df[df['name'].isin(keep)]
    keep_names = pd.DataFrame(list(keep), columns=['Names']).sort_values(by='Names')
    num = nameDF.groupby('year')['pname'].sum()
    den = num.sum()
    pname = num/den
    if prior_type == 'Emperical':
        a = df.groupby('year')['n_y'].min()
        prior = pd.Series(data=(a/a.sum()).sum(), index=years)
    elif prior_type == 'Uniform':
        prior = pd.Series(data=np.repeat(1/112,112),index=years)
    elif prior_type == 'Normal Low Certainty':
        rv = norm(loc=1980, scale=50)
        prior = pd.Series(data = rv.pdf(years)/(rv.pdf(years).sum()), index=years)
    elif prior_type == 'Normal High Certainty':
        rv = norm(loc=1980, scale=20)
        prior = pd.Series(data = rv.pdf(years)/(rv.pdf(years).sum()), index=years)
    else:
        pass
    compute = pd.concat([pname, prior], axis=1).sort_index().fillna(0) 
    cc = compute.iloc[:,0]*compute.iloc[:,1]
    post = cc/cc.sum()
    post = pd.Series(data=post, index=years)
    est = post.idxmax()

    if nameDF.shape[0] > 0:
        xx = nameDF.groupby('year')['n_x'].sum()
        fig2 = px.line(x=xx.index, y=xx, title='Observed Trend')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write('No names found.  Try again.')
    
    with st.expander("Names Included"):
        st.table(keep_names)

    if st.button('Compute Posterior'):
        fig = px.line(x=years, y=post, 
        title=f'Posterior: The most probable year of birth is {est}')
        fig.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig, use_container_width=True)
        #st.write(f'The most probably year of birth is {est}')