import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(layout='wide')
st.title("Video Stats Analysis")

st.sidebar.title('Filters')
file = st.sidebar.file_uploader('Upload Excel File (with multiple sheets):')

def filter_first_60_seconds(retention_df):
    filtered_df = retention_df.groupby('Video Title').apply(lambda x: x[x['Second'] <= 60]).reset_index(drop=True)
    return filtered_df

def filter_positions(df):
    return df[~df['Video position (%)'].isna() & (df['Video position (%)'] != '')]

def time_to_minutes(time_str):
    """Convert time string HH:MM:SS or MM:SS to minutes."""
    if pd.isna(time_str):
        return 0
    # Convert to string if it's not already
    time_str = str(time_str)
    parts = time_str.split(':')
    if len(parts) == 3:
        # Format HH:MM:SS
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        total_minutes = hours * 60 + minutes + seconds / 60
    elif len(parts) == 2:
        # Format MM:SS
        minutes = int(parts[0])
        seconds = int(parts[1])
        total_minutes = minutes + seconds / 60
    else:
        return 0
    return total_minutes

def time_to_seconds(time_str):
    """Convert time string HH:MM:SS or MM:SS to minutes."""
    if pd.isna(time_str):
        return 0
    # Convert to string if it's not already
    time_str = str(time_str)
    parts = time_str.split(':')
    if len(parts) == 3:
        # Format HH:MM:SS
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        total_seconds = hours * 3660 + minutes * 60 + seconds
    elif len(parts) == 2:
        # Format MM:SS
        minutes = int(parts[0])
        seconds = int(parts[1])
        total_seconds = minutes * 60 + seconds
    else:
        return 0
    return total_seconds

def calculate_duration(df):
    # Ensure 'Video Start' is a string
    df['Video Start'] = df['Video Start'].astype(str)
    
    # Extract maximum 'Video Start' value
    max_duration_str = df['Video Start'].max()
    
    # Split by ':' and convert to total minutes
    time_parts = max_duration_str.split(':')
    
    # Initialize total duration in minutes
    total_duration_minutes = 0
    
    if len(time_parts) == 3:  # Format HH:MM:SS
        hours, minutes, seconds = map(int, time_parts)
        total_duration_minutes = hours * 60 + minutes + seconds / 60
    elif len(time_parts) == 2:  # Format MM:SS
        minutes, seconds = map(int, time_parts)
        total_duration_minutes = minutes + seconds / 60
    
    # Set the 'Duration' column to the calculated total duration
    df['Duration'] = total_duration_minutes
    
    return df

def process_data(df, sheet_name):
    # Separate data into return and new viewers
    return_viewers = df.iloc[:, :6].copy()
    new_viewers = df.iloc[:, 6:12].copy()

    # Rename columns for new viewers to match return viewers' columns
    new_viewers.columns = return_viewers.columns

    # Add 'ViewerType' column
    return_viewers['ViewerType'] = 'return'
    new_viewers['ViewerType'] = 'new'

    # Add 'Title' column
    return_viewers['Title'] = sheet_name
    new_viewers['Title'] = sheet_name

    # Concatenate return and new viewers data
    processed_df = pd.concat([return_viewers, new_viewers], ignore_index=True)

    # Remove rows where 'Video position (%)' is NaN or 0
    processed_df = processed_df[processed_df['Video position (%)'].notna()]

    print("1.12: Printing New Concatenated df in process data", processed_df)

    # Add remaining columns from the original DataFrame
    remaining_columns = df.columns[12:]
    for col in remaining_columns:
        processed_df[col] = df[col].repeat(2).reset_index(drop=True)

    # Rename 'Return Decline' and 'New Decline' columns to 'Decline %'
    processed_df.rename(columns={'Return Decline (%)': 'Decline %', 'New Decline': 'Decline %'}, inplace=True)
    columns_to_drop = ['ChatGPT4', 'Content', 'Visuals', 'Audio','Lessons']
    columns_existing = [col for col in columns_to_drop if col in processed_df.columns]
    processed_df = processed_df.drop(columns=columns_existing)


    # Ensure all specified columns are present
    # required_columns = ['Return viewers', 'ChatGPT4', 'Content', 'Visuals', 'Audio', 'Lessons', 
    #                     'Retention 10', 'Retention 30', 'Rank Retention 30', 'Dips', 
    #                     'Flat line areas', 'Decline Areas', 'Decline %']
    # for col in required_columns:
    #     if col not in processed_df.columns:
    #         processed_df[col] = None

    # Add duration column
    video_duration = df['Video Start'].dropna().iloc[-1]  # Assuming the last non-NaN value is the total duration
    processed_df['Total Duration'] = time_to_minutes(video_duration)
    
    return processed_df

def create_multiline_chart_all(dfs, x_column, y_column, title, colors, type, dnf=False):
    fig = go.Figure()

    for i, (video_title, df) in enumerate(dfs.items()):
        print(f"Creating all video chart for : {video_title}")

        # Ensure the DataFrame contains the necessary columns
        if x_column not in df.columns or y_column not in df.columns:
            continue

        # Print the data to be plotted
        print(f"All chart - Data to be plotted:\n{df[[x_column, y_column]]}")
        print("Type is", type)

        # Check if 'ViewerType' column is present
        if 'ViewerType' in df.columns:
            for viewer_type, viewer_color in [('return', 'blue'), ('new', 'green')]:
                viewer_df = df[df['ViewerType'] == viewer_type]
                if viewer_df.empty:
                    continue
                
                fig.add_trace(go.Scatter(
                    x=viewer_df[x_column],
                    y=viewer_df[y_column],
                    mode='lines',
                    name=f"{video_title} - {viewer_type}",
                    line=dict(color=colors[i % len(colors)])
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df[x_column],
                y=df[y_column],
                mode='lines',
                name=video_title,
                line=dict(color=colors[i % len(colors)])
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend=dict(orientation="h", y=-0.55),
        xaxis_rangeslider_visible=True,
        height=600
    )
    return fig

def create_multiline_chart(dfs, x_column, y_column, title, colors, dnf=False):

    return_viewer_color = 'blue'
    new_viewer_color = 'green'

    fig = go.Figure()
    
    for i, (video_title, df) in enumerate(dfs.items()):
        print(f"Creating single chart for video: {video_title}")
        
        # Ensure the DataFrame contains the necessary columns
        if x_column not in df.columns or y_column not in df.columns:
            continue

        # Print the data to be plotted
        if 'ViewerType' in df.columns:
            print(f"Data to be plotted:\n{df[[x_column, y_column, 'ViewerType']].head(10)}")
        else:
            # In case of fig_all_videos_per_second
            print(f"Data to be plotted:\n{df[[x_column, y_column,'Retention Rate (%)']].head(10)}")

        # Plot lines for 'return' and 'new' viewers
        if 'ViewerType' in df.columns:
            for viewer_type, viewer_color in [('return', return_viewer_color), ('new', new_viewer_color)]:
                viewer_df = df[df['ViewerType'] == viewer_type]
                if viewer_df.empty:
                    continue
                
                fig.add_trace(go.Scatter(
                    x=viewer_df[x_column],
                    y=viewer_df[y_column],
                    mode='lines',
                    name=f"{video_title} - {viewer_type}",
                    line=dict(color=viewer_color)
                ))
        else:
            continue

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend=dict(orientation="h", y=-0.55),
        xaxis_rangeslider_visible=True,
        height=600
        
    )
    return fig

def create_stacked_bar_chart(dfs, x_column, y_column, title, colors):
    fig = go.Figure()
    for i, (video_title, df) in enumerate(dfs.items()):
        fig.add_trace(go.Bar(
            x=df[x_column],
            y=df[y_column],
            name=video_title,
            marker=dict(color=colors[i % len(colors)])
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        barmode='stack',
        legend=dict(orientation="h", y=-0.55),
        height=600,
        xaxis_rangeslider_visible=True
    )
    return fig

def calculate_retention_rate_per_sec(filtered_dfs):
    retention_data = []

    for video_title, df in filtered_dfs.items():
        # Process data for new and return viewers separately
        for viewer_type in ['new', 'return']:
            viewer_df = df[df['ViewerType'] == viewer_type]
            if viewer_df.empty:
                continue

            # Use the first 10 rows for simplicity
            viewer_df = viewer_df.head(10)
            
            # Convert time columns to total seconds
            viewer_df['Video Start Sec'] = viewer_df['Video Start'].astype(str).apply(time_to_seconds)
            viewer_df['Video End Sec'] = viewer_df['Video End'].astype(str).apply(time_to_seconds)

            viewer_df['Video Start Sec'] = viewer_df['Video Start Sec'].astype(float)
            viewer_df['Video End Sec'] = viewer_df['Video End Sec'].astype(float)
            
            # Filter out rows where 'Retention Start (%)' is not numeric
            viewer_df = viewer_df[pd.to_numeric(viewer_df['Retention Start (%)'], errors='coerce').notnull()]
            viewer_df['Retention Start (%)'] = viewer_df['Retention Start (%)'].astype(float)

            # Stop processing after the first occurrence where 'Video End Sec' >= 60
            if not viewer_df[viewer_df['Video End Sec'] >= 60].empty:
                viewer_df = viewer_df[viewer_df['Video End Sec'] <= 60]

            last_retention_rate = 100  # Initialize to 100% at the beginning
            last_decline_rate = 0

            # Loop running for each segment duration
            for i in range(len(viewer_df)):
                start_row = viewer_df.iloc[i]
                start_sec = int(start_row['Video Start Sec'])
                end_sec = int(start_row['Video End Sec'])
                retention_start = last_retention_rate
                retention_end = float(start_row['Retention End (%)'])
                video_position_duration = end_sec - start_sec
                decline_rate_i = start_row['Decline %']
                per_sec_change_i = decline_rate_i / video_position_duration if video_position_duration > 0 else 0
                
                # Loop running for seconds inside the segment duration to add data for each second
                for sec in range(start_sec, end_sec):
                    if sec == 0:
                        decline_rate = 0
                        per_sec_change = 0
                    else:
                        decline_rate = decline_rate_i
                        per_sec_change = per_sec_change_i

                    retention_rate = retention_start - (sec - start_sec) * per_sec_change
                    retention_data.append({
                        'Video Title': video_title,
                        'Second': sec,
                        'Video Start': start_row['Video Start'],
                        'Retention Start (%)': retention_rate,
                        'Decline %': decline_rate,
                        'Per second change': per_sec_change,
                        'ViewerType': viewer_type  # Include ViewerType
                    })

                # Update last retention rate for next segment
                last_retention_rate = retention_end

    # Create a DataFrame from the retention data
    retention_df = pd.DataFrame(retention_data)
    retention_df = retention_df.sort_values(by=['Second']).reset_index(drop=True)
    
    return retention_df



if file:

    print("1. Debugging starting")
    # Read all sheets into a dictionary of dataframes
    sheets = pd.read_excel(file, sheet_name=None)
    print("1. Reading excel files")
    # Process each sheet
    dfs = {sheet_name: process_data(df, sheet_name) for sheet_name, df in sheets.items()}

    print("1.2 Printing excel files",dfs)

    # # Create a dictionary to store return viewers count for filtering
    return_viewers_counts = {sheet_name: df['Return viewers'].sum() for sheet_name, df in dfs.items()}

    # Dictionary to store the first unique value for 'Return viewers' for each sheet

    return_viewers_counts = {sheet_name: df['Return viewers'].dropna().unique()[0] for sheet_name, df in dfs.items()}
    new_viewers_counts = {sheet_name: df['New viewers'].dropna().unique()[0] for sheet_name, df in dfs.items()}

    # Create a dictionary to store duration for filtering
    durations = {sheet_name: df['Total Duration'].max() for sheet_name, df in dfs.items()}
    
    print("1.3. Read df, duration,return viewers count",return_viewers_counts,durations)
    # Get video titles from the sheet names
    video_titles = list(dfs.keys())

    print("1.4. Printing video title",video_titles)

# Working on filters

    print("2 Working on filters now",video_titles)

    # Filter by Return viewers
    all_return_viewers = [return_viewers_counts.get(title, 0) for title in video_titles]
    min_return_viewers = int(min(all_return_viewers, default=0))
    max_return_viewers = int(max(all_return_viewers, default=1000))  # Default to 1000 if max is 0
    print("2.1 all_return_viewers,min_return_viewers,min_return_viewers",all_return_viewers,min_return_viewers,max_return_viewers)

    # Slider for Return viewers
    if min_return_viewers == max_return_viewers:
        # Ensure a valid slider range
        min_return_viewers = 0
        max_return_viewers = 1000

    selected_min_views, selected_max_views = st.sidebar.slider(
        'Select Range of Return Viewers:',
        min_value=min_return_viewers,
        max_value=max_return_viewers,
        value=(min_return_viewers, max_return_viewers)
    )
    
    # Filter by New viewers
    all_new_viewers = [new_viewers_counts.get(title, 0) for title in video_titles]
    min_new_viewers = int(min(all_new_viewers, default=0))
    max_new_viewers = int(max(all_new_viewers, default=1000))  # Default to 1000 if max is 0
    print("2.1 all_return_viewers,min_return_viewers,min_return_viewers",all_new_viewers,min_new_viewers,max_new_viewers)

    # Slider for Return viewers
    if min_new_viewers == max_new_viewers:
        # Ensure a valid slider range
        min_new_viewers = 0
        max_new_viewers = 1000

    selected_min_views_new, selected_max_views_new = st.sidebar.slider(
        'Select Range of New Viewers:',
        min_value=min_new_viewers,
        max_value=max_new_viewers,
        value=(min_new_viewers, max_new_viewers)
    )

    print("2.2 Selected new viewers",selected_min_views,selected_max_views)

    # Filter by Duration
    all_durations = [durations.get(title, 0) for title in video_titles]
    min_duration = int(min(all_durations, default=0))
    max_duration = math.ceil(max(all_durations, default=1000)) # Default to 1000 if max is 0
    print('All duration,min_duration,max_duration',all_durations,min_duration,max_duration)
    # Slider for Duration
    if min_duration == max_duration:
        # Ensure a valid slider range
        min_duration = 0
        max_duration = 1000

    selected_min_duration, selected_max_duration = st.sidebar.slider(
        'Select Range of Video Duration (minutes):',
        min_value=min_duration,
        max_value=max_duration,
        value=(min_duration, max_duration)
    )

    # Filter the videos based on the selected ranges
    filtered_video_titles = [
        title for title in video_titles 
        if selected_min_views <= return_viewers_counts.get(title, 0) <= selected_max_views and
           selected_min_duration <= durations.get(title, 0) <= selected_max_duration and
           selected_min_views_new <= new_viewers_counts.get(title, 0) <= selected_max_views_new

    ]

    # Video selection filter
    selected_video = st.sidebar.selectbox('Select Video:', options=filtered_video_titles)

    # Filter for Viewer Type
    viewer_type = st.sidebar.selectbox('Select Viewer Type:', options=['All', 'return', 'new'])

    # Show or hide decline and flat areas
    # dnf = st.sidebar.checkbox("Show Decline and Flats")
    dnf=False

    # Get the dataframe for the selected video
    df = dfs[selected_video]
    print("3.1 Dataframe for selected video",df)

    # Filter by Viewer Type if not 'All'
    if viewer_type != 'All':
        df = df[df['ViewerType'] == viewer_type]

    # ----------------------------------------------------------------------------------------------------------------  # Filter Panel Above

    # Create charts
    colors = ['#d32f2f', '#2196f3', '#e91e63', '#42a5f5', '#f06292', '#64b5f6', '#f48fb1', '#90caf9', '#ef9a9a', '#1f77b4']

    fig = create_multiline_chart({selected_video: df}, 'Video position (%)', 'Retention Start (%)', 'User Retention Chart', colors, dnf)
    fig11 = create_multiline_chart({selected_video: df}, 'Video Start', 'Retention Start (%)', 'User Retention Chart by Duration', colors)
    fig2 = create_stacked_bar_chart({selected_video: df}, 'Video position (%)', 'Decline %', 'Audience Decline Per Position Graph', colors)
    fig22 = create_stacked_bar_chart({selected_video: df}, 'Video Start', 'Decline %', 'Audience Decline Per Position Graph by Duration', colors)

    # Multiline chart for all videos
    filtered_dfs = {title: dfs[title] for title in filtered_video_titles}
    if viewer_type != 'All':
        filtered_dfs = {title: df[df['ViewerType'] == viewer_type] for title, df in filtered_dfs.items()}

    filtered_dfs_per_sec = calculate_retention_rate_per_sec(filtered_dfs)
    filtered_dfs_per_sec = filter_first_60_seconds(filtered_dfs_per_sec)
    filtered_dfs_per_sec_dict = {title: filtered_dfs_per_sec[filtered_dfs_per_sec['Video Title'] == title] for title in filtered_video_titles}

    # st.subheader('filtered_dfs Data')
    # st.write(filtered_dfs, height=200)
    # st.subheader('filtered_dfs per_sec Data')
    # st.dataframe(filtered_dfs_per_sec, height=200)
    # st.subheader('filtered_dfs per_sec dict Data')

        
        
    fig_all_videos = create_multiline_chart_all(filtered_dfs, 'Video position (%)', 'Retention Start (%)', 'User Retention Chart for All Videos by Video Position', colors,"Position",                               dnf)
    fig_all_videos_per_second = create_multiline_chart_all(filtered_dfs_per_sec_dict, 'Second', 'Retention Start (%)', 'User Retention Chart for All Videos by Video Duration', colors,"Duration", dnf)
    
    # Printing the number of return viewers for the particular video
 
    col1, col2 = st.columns((4, 2))
    with col1:
        col11, col12 = st.columns((3, 1))
        with col11:
            gs1 = st.selectbox('Select Chart', options=['User Retention Chart', 'Audience Decline Per Position Graph'])
        with col12:
            gs12 = st.selectbox('Select By:', options=['By Video Position', 'By Duration'])
        if gs1 == 'Audience Decline Per Position Graph' and gs12 == 'By Video Position':
            st.plotly_chart(fig2, use_container_width=True)
        elif gs1 == 'User Retention Chart' and gs12 == 'By Duration':
            st.plotly_chart(fig11, use_container_width=True)
        elif gs1 == 'Audience Decline Per Position Graph' and gs12 == 'By Duration':
            st.plotly_chart(fig22, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
    
    # Show the multiline chart for all videos
    # return_viewers_value = df['Return viewers'].iloc[0]
    # new_viewers_value = df['New viewers'].iloc[0]
    # st.subheader(f'Return Viewers: {return_viewers_value}')

    try:
        return_viewers_value = df['Return viewers'].iloc[0]
        new_viewers_value = df['New viewers'].iloc[0]

    # Display the values beneath the chart with improved styling
        st.markdown(f"""
    <style>
    .viewer-stats {{
        font-size: 1.1em;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
    }}
    .viewer-stats .value {{
        color: #007BFF;
    }}
    </style>
    <div class="viewer-stats">
        <p>Return Viewers: <span class="value">{return_viewers_value}</span></p>
        <p>New Viewers: <span class="value">{new_viewers_value}</span></p>
    </div>
""", unsafe_allow_html=True)
    
    except:
        None

    st.subheader('User Retention Chart for All Videos')
    st.plotly_chart(fig_all_videos, use_container_width=True)

    st.plotly_chart(fig_all_videos_per_second, use_container_width=True)

    # Display processed data in a scrollable table at the bottom
    st.subheader('Processed Data')
    st.dataframe(df, height=200)

    for video_title, df in filtered_dfs_per_sec_dict.items():
        if selected_video==video_title:
            st.subheader(f"Per sec Retention Data for {video_title}")
            st.dataframe(df, height=200)  # Adjust height as needed
        else:
            continue

else:
    st.info("Upload the excel sheet first to continue")
