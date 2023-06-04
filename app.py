import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 =  open("pipeline.pkl", "rb")
rf = pickle.load(file1)
file1.close()

# Company,TypeName,Ram,OpSys,Weight,Touchscreen,IPS,PPI,CPU_name,HDD,SSD,Gpu brand
# Apple,Ultrabook,8,Mac,1.37,0,1,226.98300468106115,Intel Core i5,0,128,Intel

data = pd.read_csv("cleaned_X_train_data.csv")

data["IPS"].unique()

with st.container():
        st.title("Make The Price Of That Laptop💻 No Shock You!")
        
        st.write("##")
        st.write("""
                Imagine never again having to worry about walking into a store and realizing 
                that the laptop you need is way beyond your budget. \n 
                Our project aims to solve this common 
                challenge faced by tech enthusiasts and laptop buyers alike. \n
                By leveraging the power of data science, 
                we have developed an intelligent system that provides estimated laptop prices based on desired specifications.\n 
                With this innovative tool, you can confidently enter a store, knowing exactly how much you need to spend. 
                Say goodbye to surprises and hello to a seamless and informed laptop buying experience.
                """)
        st.write("##")
        st.header("Problem Solving Steps")
        st.write("- Loading the Data into Pandas Data frame")
        st.write("- Data Preprocessing and Visualization")
        st.write("- Model Building (selecting the best Algorithm for prediction)")
        st.write("- Hyperparameter Tuning (setting the right parameters for the algorithm)")
        st.write("- Saving the model into a Pickle File and Integrating it with UI")
        st.write("- Deploying the Application locally.")
        st.write("##")
        st.write("##")


        if st.button("View Project Source Code"):
            "(https://github.com/daniel-datasci/Laptop-Price-Predictor-By-Daniel)"
        st.write("##")
        st.write("##")
        st.header("Application Area")
        st.markdown("""
                    All Fields Are Required To For The Application To Work Correctly
                    """)

        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            # Company
            company = st.selectbox("Brand", data["Company"].unique())

        with middle_column:
            # Type of Laptop
            type = st.selectbox("Type", data["TypeName"].unique())

        with right_column:
            # Ram present in Laptop
            ram = st.selectbox("Ram (in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            # OS of laptop
            os = st.selectbox("OS", data["OpSys"].unique())
        with middle_column:
            # weight of laptop
            weight = st.number_input("Weight of the laptop")
        with right_column:
            # Touchscreen available in laptop or not
            touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            # IPS
            ips = st.selectbox('IPS', ['No', 'Yes'])
        with middle_column:
            # screen size
            screen_size = st.number_input('Screen Size')
        with right_column:
            # resolution of laptop
            resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
        left_column, right_column = st.columns(2)
        with left_column:
            # cpu
            cpu = st.selectbox('CPU', data['CPU_name'].unique())
        with right_column:
            # hdd
            hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
        left_column, right_column = st.columns(2)
        with left_column:
            # ssd
            ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
        with right_column:
            # Gpu
            gpu = st.selectbox('GPU(in GB)', data['Gpu brand'].unique())

        if st.button("Predict Price"):

            ppi = None
            if touchscreen == "Yes":
                touchscreen = 1
            else:
                touchscreen = 0

            if ips == "Yes":
                ips = 1
            else:
                ips = 0

            X_resolution = int(resolution.split("x")[0])
            Y_resolution = int(resolution.split("x")[1])

            ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(int(screen_size))

            query = np.array([company, type, ram,  os, float(weight),
                            touchscreen, ips, ppi, cpu, hdd, ssd, gpu,])

            query = query.reshape(1, 12)

            prediction = int(np.exp(rf.predict(query)[0]))

            st.title('Estimated price for this laptop could be between ' + '₦' + str(round((prediction-12.1)*750)) + ' ' + ' to ' + ' ' + '₦' + str(round((prediction+12.1)*750)))
