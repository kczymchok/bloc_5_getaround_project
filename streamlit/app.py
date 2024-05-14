import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests
# import os
import pandas as pd
import numpy as np

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.subplots as sp
from plotly.subplots import make_subplots
import json



### Config
st.set_page_config(page_title="VroumVroum", page_icon="", layout="wide")

delay_df = pd.read_excel('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx',sheet_name='rentals_data')
pricing = pd.read_csv('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv',index_col=0)
pricing = pricing[pricing['mileage'] >= 0]
pricing = pricing[pricing['engine_power'] > 0]


# ---------------------------------------   MENU DEROULANT    --------------------------------------   

if __name__ == '__main__': 

    with st.sidebar:
        selected = option_menu("Menu", ["Home", 'Goals','Delays EDA','Prices EDA','API'], 
            icons=['house', 'archive','alarm','activity','app-indicator'], menu_icon="car-front-fill", default_index=0)
        selected

# ---------------------------------------   HOME    --------------------------------------   

    if selected == 'Home':
        st.write("<h1 style='text-align:center; color: purple;'> GETAROUND PROJECT </h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: white;'>Behind the Wheel : Exploring Car Rental Data for Strategic Insights</h1>", unsafe_allow_html=True)
        # st.markdown("<h3 style='text-align: center; color: white;'> The new Airbnb for cars ! </h3>", unsafe_allow_html=True)
        # st.markdown("---")

        st.markdown("---")
        st.write('')
        col1, col2 = st.columns([1,2])
    
        with col1:
            st.write("")
            st.subheader("Get a what ?")
            st.write("")
            st.write("""GetAround is the Airbnb for cars. You can rent cars from any person for a few hours to a few days!

Founded in 2009, this company has known rapid growth. 
In 2019, they count over 5 million users and about 20K available cars worldwide. 

As Jedha's partner, they offered this great challenges:""")
            st_lottie("https://assets9.lottiefiles.com/packages/lf20_eP48EC.json",key='tuture')

        with col2:
            st.write("")
            st.subheader("Context")
            st.write("")
            st.write("""When renting a car, our users have to complete a checkin flow at the beginning of the rental and a checkout flow at the end of the rental in order to:
- Assess the state of the car and notify other parties of pre-existing damages or damages that occurred during the rental.
- Compare fuel levels.
- Measure how many kilometers were driven.

The checkin and checkout of our rentals can be done with three distinct flows:

- 📱 Mobile rental agreement on native apps: driver and owner meet and both sign the rental agreement on the owner’s smartphone
- Connect: the driver doesn’t meet the owner and opens the car with his smartphone
- 📝 Paper contract (negligible)""")

        st.markdown("---")
        st.write("")
        st.subheader("Project ")
        st.write("")
        st.write("""For this case study, we suggest that you put yourselves in our shoes, and run an analysis we made back in 2017 🔮 🪄

When using Getaround, drivers book cars for a specific time period, from an hour to a few days long. They are supposed to bring back the car on time, but it happens from time to time that drivers are late for the checkout.

Late returns at checkout can generate high friction for the next driver if the car was supposed to be rented again on the same day : Customer service often reports users unsatisfied because they had to wait for the car to come back from the previous rental or users that even had to cancel their rental because the car wasnt returned on time.""")
        
        st.markdown("---")

# ---------------------------------------   PAGE DE CONSIGNES      --------------------------------------   

    if selected == 'Goals':
        st.markdown("<h1 style='text-align: center; color: white;'>Goals for the project</h1>", unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown(" ")
        col1, col2 = st.columns([3,2])
        with col1:
            st.write("""
        In order to mitigate those issues we’ve decided to implement a minimum delay between two rentals. A car won’t be displayed in the search results if the requested checkin or checkout times are too close from an already booked rental.

        It solves the late checkout issue but also potentially hurts Getaround/owners revenues: we need to find the right trade off.

        Our Product Manager still needs to decide:""") 
        with col2:
            st_lottie("https://assets9.lottiefiles.com/private_files/lf30_hsabbeks.json", key="contrat_voiture")
        col1, col2,col3,col4 = st.columns([1,1,1,1])
        with col2:
            st.subheader("""THRESHOLD: 
How long should the minimum delay be?""")
        with col3:
            st.subheader("""SCOPE:
Should we enable the feature for all cars?, only Connect cars?""")
        st.markdown(" ")
        st.write("""In order to help them make the right decision, they are asking you for some data insights. Here are the first analyses they could think of, to kickstart the discussion. Don’t hesitate to perform additional analysis that you find relevant.

- Which share of our owner’s revenue would potentially be affected by the feature?
- How many rentals would be affected by the feature depending on the threshold and scope we choose?
- How often are drivers late for the next check-in? How does it impact the next driver?
- How many problematic cases will it solve depending on the chosen threshold and scope?""")

        st.markdown("---")

        col1,col2 = st.columns([1,2])
        with col1:
            st.button('Web dashboard',key='Web dashboard')
            st.write("First build a dashboard that will help the product Management team with the above questions. You can use streamlit or any other technology that you see fit.")
            st_lottie('https://assets6.lottiefiles.com/packages/lf20_acryqbdv.json',key='dashboard')

        with col2:
            st.button('Machine Learning',key='Machine Learning')
            st.write("""
In addition to the above question, the Data Science team is working on pricing optimization. They have gathered some data to suggest optimum prices for car owners using Machine Learning.
You should provide at least one endpoint /predict. The full URL would look like something like this: https://your-url.com/predict.
This endpoint accepts POST method with JSON input data and it should return the predictions. We assume inputs will be always well formatted. It means you do not have to manage errors. We leave the error handling as a bonus.

Input example:
{
"input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8], [7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]
}
The response should be a JSON with one key prediction corresponding to the prediction.

Response example:
{
"prediction":[6,6]
}
""")
        st.markdown("---")

        col1,col2,col3 = st.columns([1,1,1])

        with col1:
            st.button('Documentation',key='Documentation')
            st.write("""You need to provide the users with a documentation about your API.
It has to be located at the /docs of your website. If we take the URL example above, it should be located directly at https://your-url.com/docs).

This small documentation should at least include:
An h1 title: the title is up to you.
A description of every endpoints the user can call with the endpoint name, the HTTP method, the required input and the expected output (you can give example).
You are free to add other any other relevant informations and style your HTML as you wish.
        """)

        with col2:
            st.button('Online Production',key='Online Production')
            st.write("You have to host your API online. We recommend you to use Heroku as it is free of charge. But you are free to choose any other hosting provider.")
            st_lottie('https://assets1.lottiefiles.com/packages/lf20_xsicerbj.json',key='online')
        with col3:
            st.button('Helpers',key='Helpers')
            st.write("""
To help you start with this project we provide you with some pieces of advice:

Spend some time understanding data
Don't overlook Data Analysis part, there is a lot of insights to find out.

Data Analysis should take 2 to 5 hours
Machine Learning should take 3 to 6 hours
You are not obligated to use libraries to handle your Machine Learning workflow like mlflow but we definitely advise you to do so.""")

        st.markdown("---")

        st.button("Delivrable",key='delivrable')
        col1,col2 = st.columns([3,1])
        with col1:
            st.write("""
In order to get evaluation, do not forget to share your code on a Github repository. You can create a README.md file with a quick description about this project, how to setup locally and the online URL.

To complete this project, you should deliver:
A dashboard in production (accessible via a web page for example)
The whole code stored in a Github repository. You will include the repository's URL.
An documented online API on Heroku server (or any other provider you choose) containing at least one /predict endpoint that respects the technical description above. We should be able to request the API endpoint /predict using curl or Python.""")
        with col2:
            st_lottie('https://assets1.lottiefiles.com/packages/lf20_6ft9bypa.json',key='exam')

# ---------------------------------------   EDA DELAYS      --------------------------------------   

    if selected == "Delays EDA":
        st.markdown("<h1 style='text-align: center; color: white;'>Exploratory Data Analysis - DELAYS</h1>", unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown(" ")


        st.markdown("<h2 style='text-align: center; color: white;'>Overview of delays</h2>", unsafe_allow_html=True)

        st.write("The most important variable for us is the delay, which has missing values. We need to remove them.")
        delay_df.dropna(subset=['delay_at_checkout_in_minutes'],inplace=True)
        st.write(f"We remove the variable 'state', since there is only {len(delay_df[delay_df['state']=='canceled'])} canceled rental and the rest is all normal ended rentals.")
        delay_df.drop(columns="state")
        positive_delay_df = delay_df[delay_df['delay_at_checkout_in_minutes']>0]

        # We cannot use make_subplot in streamlit, and we cannot use sp.subplots for pies. 
        # We need to make separate pies though.

        # All users
        bins = [-float('inf'), 0, float('inf')]
        labels = ['Users on time or in advance', 'Delayed users']
        delay_bins = pd.cut(delay_df.delay_at_checkout_in_minutes, bins=bins, labels=labels)
        section_counts = delay_bins.value_counts()
        # Positive delays only
        bins = [0, 60, 120, 240, float('inf')]
        labels = ['0 to 1h', '1 to 2h', '2h to 4h','>7h']
        delay_bins_positive = pd.cut(positive_delay_df.delay_at_checkout_in_minutes, bins=bins, labels=labels)
        section_counts_positive = delay_bins_positive.value_counts()
        # Subplots & graphs
        specs = [[{'type': 'domain'}, {'type': 'domain'}]]
        fig = make_subplots(rows=1, cols=2, specs=specs, subplot_titles=['There are more delayed users than users on time','Most delayed users have less than 2h delay'])
        fig.add_trace(go.Pie(labels=section_counts.index, values=section_counts, textinfo='label+percent'),row=1, col=1)
        fig.add_trace(go.Pie(labels=section_counts_positive.index, values=section_counts_positive, textinfo='label+percent'),row=1, col=2)
        fig.update_layout(title='Repartition of delay at checkout', title_font=dict(size=20), showlegend=False)
        st.plotly_chart(fig)

        mean_delay = round(delay_df['delay_at_checkout_in_minutes'].mean())
        st.write(f"The average delay is of {mean_delay} minutes ({mean_delay/60} hours).")
        average_delay_for_positive_delay = round(positive_delay_df.delay_at_checkout_in_minutes.mean())
        median_delay_for_positive_delay = round(positive_delay_df.delay_at_checkout_in_minutes.median())
        st.write(f"The average delay of delayed people is of {average_delay_for_positive_delay} minutes ({round(average_delay_for_positive_delay/60)} hours).")
        st.write(f"The median is quite different because of extremes values: {median_delay_for_positive_delay} minutes ({round(median_delay_for_positive_delay/60)} hour).")

        # Create subplots
        fig = sp.make_subplots(rows=1, cols=3, subplot_titles=("Delay is extremely spread", "Users have 60min delay in average","Positive delay: 3h avg vs 1h median"))
        # First graph : Boxplot of delays
        boxplot = go.Box(y=delay_df['delay_at_checkout_in_minutes'],showlegend=False, marker=dict(color='blue'), name='')
        fig.add_trace(boxplot, row=1, col=1)
        fig.update_yaxes(title_text="Delay (minutes)", row=1, col=1)
        # Second graph : Histogram of delays
        histogram1 = go.Histogram(x=delay_df['delay_at_checkout_in_minutes'],showlegend=False, marker=dict(color='blue') )
        fig.add_trace(histogram1, row=1, col=2)
        fig.update_xaxes(title_text="Delay (minutes) zoomed in", range=[-400, 400], row=1, col=2)
        # Add average and its legend
        fig.add_shape(type="line",x0=mean_delay,y0=0,x1=mean_delay,y1=5000,line=dict(color="red", width=2, dash="dash"),row=1,col=2,)
        fig.add_trace(go.Scatter(x=[mean_delay], y=[0], mode="lines", name="Average delay all users (1h)", line=dict(color="red", width=2, dash="dash")), row=1, col=2)
        # Third graph : Histogram of positive delays
        histogram2 = go.Histogram(x=positive_delay_df['delay_at_checkout_in_minutes'],showlegend=False, marker=dict(color='blue'))
        fig.add_trace(histogram2, row=1, col=3)
        fig.update_xaxes(title_text="Delay (minutes) zoomed in", range=[0, 1000], row=1, col=3)
        # Add average and its legend
        fig.add_shape(type="line",x0=average_delay_for_positive_delay,y0=0,x1=average_delay_for_positive_delay,y1=7000,line=dict(color="green", width=2, dash="dash"),row=1,col=3)
        fig.add_trace(go.Scatter(x=[average_delay_for_positive_delay], y=[0], mode="lines", name="Average positive delays (3h)", line=dict(color="green", width=2, dash="dash")), row=1, col=3)
        # Add median and its legend
        fig.add_shape(type="line",x0=median_delay_for_positive_delay,y0=0,x1=median_delay_for_positive_delay,y1=7000,line=dict(color="yellow", width=2, dash="dash"),row=1,col=3)
        fig.add_trace(go.Scatter(x=[median_delay_for_positive_delay], y=[0], mode="lines", name="Median positive delays (1h)", line=dict(color="yellow", width=2, dash="dash")), row=1, col=3)
        fig.update_layout(title="Delay at Checkout",title_font=dict(size=20),legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),width=800)
        st.plotly_chart(fig)


        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>Impact of delays on next users</h2>", unsafe_allow_html=True)

        # DATA PREPARATION

        sorted_delay = delay_df.sort_values(by=['car_id', 'rental_id'])
        sorted_delay.head(100)
        # Create the column where we will put delay of the previous users:
        # delay['delay_of_previous_user'] = np.nan
        # For everyline, we look if there is a value in "previous_ended_rental_id" column:
        for index, row in sorted_delay.iterrows():
            previous_ended_rental_id = row['previous_ended_rental_id']
            if pd.notnull(previous_ended_rental_id):
        # if there is a value, we look for the corresponding "rental_id" in the table and retrieve the delay associated:
                previous_delay = sorted_delay.loc[sorted_delay['rental_id'] == previous_ended_rental_id, 'delay_at_checkout_in_minutes']
                if not previous_delay.empty:
                    sorted_delay.at[index, 'delay_of_previous_user'] = previous_delay.iloc[0]
        # Now we calculate the difference between the expected timegap between 2 rentals, and the delay of the 1st one.
        sorted_delay['delta_timegap_delay'] = sorted_delay['time_delta_with_previous_rental_in_minutes'] - sorted_delay['delay_of_previous_user']
        # Show table
        pd.set_option('display.max_rows', None)
        sorted_delay.head(10)
        # Check table: for car 159533, we see the delay has indeed been reported.

        # OVERVIEW OF TIMEGAPS

        # Analyzing time_delta_with_previous_rental_in_minutes
        st.write("Timegap is the time expected between 2 rentals.")
        df_timegaps = sorted_delay.dropna(subset='time_delta_with_previous_rental_in_minutes')
        st.write(f"This variable has only {len(df_timegaps)} values, so we create a new dataset df_timegaps")
        average_time_gap = df_timegaps['time_delta_with_previous_rental_in_minutes'].mean()
        # Distribution of time gaps
        fig = px.histogram(df_timegaps, x='time_delta_with_previous_rental_in_minutes')
        # Add the average line
        fig.add_trace(go.Scatter(x=[average_time_gap, average_time_gap],y=[0, 350],mode='lines',line=dict(color='red', dash='dash'),name='Average (5h)'))
        fig.update_layout(title="Distribution of Time Gaps planned between Consecutive Rentals",xaxis_title="Time Gap (minutes)",yaxis_title="Count",showlegend=True)
        st.plotly_chart(fig)
        st.write(f"In average, there is {round(average_time_gap / 60)}h time gap between consecutive rentals.")
        # Short turnaround times:
        short_turnaround = df_timegaps[df_timegaps['time_delta_with_previous_rental_in_minutes'] < 60]
        percentage_short_turnaround = 100 * len(short_turnaround) / len(df_timegaps)
        st.write(f"The most encountered situation is a short turnaround (less than 1h): {round(percentage_short_turnaround)}% of rentals with a timegap.")

        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>Problematic cases</h2>", unsafe_allow_html=True)


        st.write('Time gaps is the time expected between 2 planned rentals (check out and check in).')
        st.write("A problematic case would be when a car is delayed more than the expected timegap before the next rental.")
        # Distribution of delta (difference between the timegap and the previous user's delay)
        fig = px.histogram(df_timegaps, x='delta_timegap_delay',color='state')
        # Add the average line
        average_delta_timegap_delay = df_timegaps.delta_timegap_delay.mean()
        fig.add_trace(go.Scatter(x=[average_delta_timegap_delay, average_delta_timegap_delay],y=[0, 350],mode='lines',line=dict(color='yellow', dash='dash'),name=f'Average ({round(average_delta_timegap_delay/60)}h)'))
        # Update layout
        fig.update_layout(title="Distribution of delta timegap minus delay",title_font=dict(size=20),xaxis_title="Delta (Timegap - Previous user's delay) in minutes",yaxis_title="Count",showlegend=True)
        # Zoom in on x-axis
        fig.update_xaxes(range=[-2000, 2500])
        st.plotly_chart(fig)
        st.write(f"The difference between timegap and previous user's delay is generaly positive, which means not problematic.")
        st.write(f"The average of this delta is {round(average_delta_timegap_delay/60)}h.")
        
        # Exploration of problematic cases

        problematic_cases = df_timegaps[df_timegaps['delta_timegap_delay']<0]
        percentage_problematic_cases_from_timegaps = round (100 * len(problematic_cases) / len(df_timegaps))
        percentage_problematic_cases_from_all = round (100 * len(problematic_cases) / len(sorted_delay))
        st.write(f"Among {len(df_timegaps)} cases known of 2 subsequent rentals, {len(problematic_cases)} were problematic ({percentage_problematic_cases_from_timegaps}%), which is only {percentage_problematic_cases_from_all}% of all rentals.")
        st.write('Most problematic cases are due to a previous delay of the previous user of less than 2h.')
        # Proportion problematic cases
        labels_1_2 = ['Problematic', 'Not problematic']
        values1 = [len(problematic_cases), len(df_timegaps) - len(problematic_cases)]
        values2 = [len(problematic_cases), len(sorted_delay) - len(problematic_cases)]
        # Delay of previous user in problematic case
        bins = [0, 60, 120, float('inf')]
        labels_3 = ['Less than 1h', '1 to 2h', 'More than 2h']
        delay_bins = pd.cut(problematic_cases['delay_of_previous_user'], bins=bins, labels=labels_3)
        values3 = delay_bins.value_counts(sort=False)
        # Subplots
        specs = [[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]
        fig = make_subplots(rows=1, cols=3, specs=specs, subplot_titles=(f"Problematic cases represent {percentage_problematic_cases_from_timegaps}% of Time Gaps", f"which is only {percentage_problematic_cases_from_all}% of all rentals", "Delay of Previous User in Problematic Cases"))
        fig.add_trace(go.Pie(labels=labels_1_2, textinfo='label+percent', values=values1), row=1, col=1)
        fig.add_trace(go.Pie(labels=labels_1_2, textinfo='label+percent', values=values2), row=1, col=2)
        fig.add_trace(go.Pie(labels=labels_3, textinfo='label+percent',values=values3), row=1, col=3)
        fig.update_layout(title='Problematic cases', title_font=dict(size=20), showlegend=False,width=800)
        st.plotly_chart(fig)


        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>Defining a threshold</h2>", unsafe_allow_html=True)


        st.write("Most delayed people have less than 1h delay.")
        st.write("And time gaps between 2 rentals are mostly less than 1h ")
        thresholds = [60,80,100,150,200]
        columns = ['threshold','rentals lost','rentals lost (perc. from timegaps)','rentals lost (perc. from all)','pb cases','pb cases (perc. from timegaps)']

        df_thresholds = []
        for threshold in thresholds:
            rentals_left_in_timegaps = df_timegaps[df_timegaps['time_delta_with_previous_rental_in_minutes'] > threshold]
            rentals_lost = len(df_timegaps) - len(rentals_left_in_timegaps)
            percentage_rentals_lost_in_timegaps = round(100 * rentals_lost / len(df_timegaps))
            percentage_rentals_lost_in_total = round(100 * rentals_lost / len(sorted_delay))
            df_problematic_cases = rentals_left_in_timegaps[rentals_left_in_timegaps['delta_timegap_delay'] < 0]
            problematic_cases_count = len(df_problematic_cases)
            percentage_problematic_cases_in_timegaps = round(100 * problematic_cases_count / len(df_timegaps), 1)
            data = {
                'threshold': threshold,
                'rentals lost': rentals_lost,
                'rentals lost (perc. from timegaps)': percentage_rentals_lost_in_timegaps,
                'rentals lost (perc. from all)': percentage_rentals_lost_in_total,
                'pb cases': problematic_cases_count,
                'pb cases (perc. from timegaps)': percentage_problematic_cases_in_timegaps
            }
            df_thresholds.append(data)
        df_thresholds = pd.DataFrame(df_thresholds,columns=columns)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        st.write(df_thresholds)
        st.write(f"If we choose 80 minutes threshold of timegap minimum between 2 rentals, we will loose 32% of rentals with timegaps (or 3% of total rentals).")
        st.write("However, this will eradicate almost all problematic cases (2.1% of rentals with timegaps compared to 11% before putting a threshold.)")


        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>Importance of checkin type</h2>", unsafe_allow_html=True)

        fig = sp.make_subplots(rows=1, cols=3, subplot_titles=["Number of contracts by check-in type", "Delays in minutes", "Delays in minutes zoomed-in"])
        # First subplot - Histogram
        histogram = go.Histogram(x=sorted_delay['checkin_type'])
        fig.add_trace(histogram, row=1, col=1)
        # Second subplot - Box plot
        box_plot = go.Box(x=sorted_delay['checkin_type'], y=sorted_delay['delay_at_checkout_in_minutes'])
        fig.add_trace(box_plot, row=1, col=2)
        # Third subplot - Box plot with updated y-axis range
        box_plot_range = go.Box(x=sorted_delay['checkin_type'], y=sorted_delay['delay_at_checkout_in_minutes'])
        fig.add_trace(box_plot_range, row=1, col=3)
        # Update y-axis range for the third subplot
        fig.update_yaxes(range=[-300, 300], row=1, col=3)
        # Update layout with general title
        fig.update_layout(height=400, width=800, title="Impact of check-in type on delays",showlegend=False)
        st.plotly_chart(fig)
        st.write("There are more people subscribing by mobile than by connect app.")
        st.write("The delay for checkout is very spread for contract by mobile than the Connect app.")
        st.write("Generally, people do the checkout late when signing via mobile, and give back the car in advance when using the connect app.")
        
        # Impact of type of checkin on problematic cases
        checkin_type_counts = problematic_cases['checkin_type'].value_counts()
        fig = go.Figure(data=go.Pie(labels=checkin_type_counts.index, values=checkin_type_counts.values))
        fig.update_layout(title='Repartition of Problematic Cases by Check-in Type', title_font=dict(size=20),width=600)
        st.plotly_chart(fig)
        st.write("Most problematic cases are mostly from users using the mobile checkin.")
        st.write("The scope is the following: we will apply our threshold only for people suscribing via mobile.")
        
        
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>Final Threshold/Scope </h2>", unsafe_allow_html=True)

        # Recalculate lost rentals and problematic cases solved:
        threshold = 80
        scope = 'mobile'
        rentals_lost = len(df_timegaps[df_timegaps['checkin_type'] == scope][df_timegaps['time_delta_with_previous_rental_in_minutes'] < threshold])
        percentage_rentals_lost_in_timegaps = round(100 * rentals_lost / len(df_timegaps))
        percentage_rentals_lost_in_total = round(100 * rentals_lost / len(delay_df))
        df_problematic_cases = rentals_left_in_timegaps[rentals_left_in_timegaps['delta_timegap_delay']<0]
        problematic_cases_count = len(df_problematic_cases)
        percentage_problematic_cases_in_timegaps = round(100 * problematic_cases_count / len(df_timegaps) , 1)
        st.write(f"With a threshold of {threshold}min and a scope by {scope} suscription, we get a loss of {percentage_rentals_lost_in_timegaps}% rentals in timegaps, \nso {percentage_rentals_lost_in_total}% of total rentals. This would decrease problematic cases to {percentage_problematic_cases_in_timegaps}% for rentals with timegaps (compared \nto 11% initially).")
        
        # Money lost
        pricing_average = round(pricing.rental_price_per_day.mean())
        pricing_median = round(pricing.rental_price_per_day.median())
        st.write(f"Average price per day: {pricing_average}€, Median price per day: {pricing_median}€.")
        average_money_won = round(len(delay_df) * pricing_average)
        money_lost = round((2/100) * len(delay_df) * pricing_average)
        malus = round(money_lost / ((2/100) * len(delay_df)))
        st.write(f"With the chosen threshold and scope, we would loose {money_lost}€ per day.")
        st.write(f"If we don't want to loose money, we can make a malus of {malus}€ per person delayed per day.")
        st.write(f"Even though it's not reasonable, given the price of the rental per day.")




# --------------------------------   EDA PRICING    --------------------------------   

    if selected == "Prices EDA":
        st.markdown("<h1 style='text-align: center; color: white;'>Exploratory Data Analysis - PRICING</h1>", unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown(" ")
        
        st.write("No missing value, 2 numerical (mileage, engine power), 11 categorical (4 str, 7 bool).")
        st.write("Target value is rental price per day.")


        st.markdown("<h2 style='text-align: center; color: white;'>Numerical Variables</h2>", unsafe_allow_html=True)


        fig = make_subplots(rows=1, cols=3, subplot_titles=("Mileage", "Engine Power", "Price per Day"))
        # First subplot - Mileage
        fig.add_trace(go.Histogram(x=pricing['mileage'], histnorm='density'), row=1, col=1)
        fig.update_xaxes(title_text="Mileage", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.add_shape(type='line', x0=pricing['mileage'].mean(), x1=pricing['mileage'].mean(), y0=0, y1=0.1, line=dict(color='red', dash='dash'), row=1, col=1)
        # Second subplot - Engine Power
        fig.add_trace(go.Histogram(x=pricing['engine_power'], histnorm='density'), row=1, col=2)
        fig.update_xaxes(title_text="Engine Power", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=2)
        fig.add_shape(type='line', x0=pricing['engine_power'].mean(), x1=pricing['engine_power'].mean(), y0=0, y1=200, line=dict(color='red', dash='dash'), row=1, col=2)
        # Third subplot - Price per Day
        fig.add_trace(go.Histogram(x=pricing['rental_price_per_day'], histnorm='density'), row=1, col=3)
        fig.update_xaxes(title_text="Rental Price per Day", row=1, col=3)
        fig.update_yaxes(title_text="Density", row=1, col=3)
        fig.add_shape(type='line', x0=pricing['rental_price_per_day'].mean(), x1=pricing['rental_price_per_day'].mean(), y0=0, y1=100, line=dict(color='red', dash='dash'), row=1, col=3)
        # Update layout & show
        fig.update_layout(height=400, width=900, title="Distribution of Pricing Features", showlegend=False)
        st.plotly_chart(fig)
        
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>Categorical Variables</h2>", unsafe_allow_html=True)


        pricing_categorical = pricing.drop(columns=['mileage', 'engine_power', 'rental_price_per_day'])
        fig, axes = plt.subplots(4, 3, figsize=(12, 4*3))
        # Flatten the axes array to simplify indexing
        axes = axes.flatten()
        for i, column in enumerate(pricing_categorical.columns):
            ax = axes[i]
            counts = pricing_categorical[column].value_counts()
            counts.plot(kind='bar', ax=ax)
            ax.set_title(f"Count of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
        # Hide unused subplots if any
        for j in range(11, 12):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>Correlation with target / within variables</h2>", unsafe_allow_html=True)


        # Visualize pairwise dependencies
        fig = px.scatter_matrix(pricing)
        fig.update_layout(
                title = go.layout.Title(text = "Bivariate analysis", x = 0.5), showlegend = False, 
                    autosize=False, height = 2000, width = 2000)
        st.plotly_chart(fig)
        st.write("The variables that are the most correlated with rental price are mileage (anticorrelated) and engine power.\nExcept for winter tires, all other variables seem important too.\nVariables don't seem correlated otherwise between themselves.")

        # correlations_with_target = {}
        # corr_matrix = pricing.corr().round(2)
        # for column in corr_matrix.columns:
        #     if column != 'rental_price_per_day':
        #         correlation = corr_matrix.loc[column, 'rental_price_per_day']
        #         correlations_with_target[column] = correlation
        # st.write(correlations_with_target)

        st.write("""Knowing that we have only 2 numerical variables, and they are both the most correlated with the target,
            we will do feature engineering on those variables. The square value and the logarithm of "mileage" could
            be interesting as it has a negative correlation. For engine_power, we will test the square, the cubed, the
            inverse of it, and then try multiply it with mileage and mileage_squared""")



# --------------------------------   API  PREDICTION    --------------------------------   
    if selected == 'API':
        st.markdown("<h1 style='text-align: center; color: white;'> API for prediction of rental price per day</h1>", unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown(" ")
        
        col1,col2 = st.columns([3,1])
        with col1:
            st.write("An API was created in order to predict the rental price per day of a car depending on the parameters that the user choose (color, fuel etc)")    
            api_link = st.button("API documentation link", key = "API doc")
            if api_link:
                # import webbrowser
                # webbrowser.open_new_tab("https://getaround-api-elo-16ab161d9781.herokuapp.com/docs")

                from bokeh.models.widgets import Div
                js = "window.open('https://getaround-apii-0d894b86a54e.herokuapp.com')"
                js = "window.location.href = 'https://getaround-apii-0d894b86a54e.herokuapp.com/docs'"
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)


        with col2:
            st_lottie("https://assets2.lottiefiles.com/packages/lf20_fvrs1qak.json")


# -------------------------------------     END    ---------------------------------------

        ### Footer 
        empty_space, footer = st.columns([1, 2])

        with empty_space:
            st.write("")

        with footer:
            st.write("")
            st.write("")
            st.markdown("""🚗 Thanks for your attention. 🚚	""")

        st.markdown("---")
