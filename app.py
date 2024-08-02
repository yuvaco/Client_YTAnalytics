import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout='wide')
st.title("Video Stats Analysis")

st.sidebar.title('Filters')
file = st.sidebar.file_uploader('Upload Excel File (with multiple sheets):')

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

    # Add remaining columns from the original DataFrame
    remaining_columns = df.columns[12:]
    for col in remaining_columns:
        processed_df[col] = df[col].repeat(2).reset_index(drop=True)

    # Rename 'Return Decline' and 'New Decline' columns to 'Decline %'
    if 'Return Decline (%)' in processed_df.columns:
        processed_df.rename(columns={'Return Decline (%)': 'Decline %'}, inplace=True)
    if 'New Decline' in processed_df.columns:
        processed_df.rename(columns={'New Decline': 'Decline %'}, inplace=True)

    # Ensure all specified columns are present
    required_columns = ['Return viewers', 'ChatGPT4', 'Content', 'Visuals', 'Audio', 'Lessons', 
                        'Retention 10', 'Retention 30', 'Rank Retention 30', 'Dips', 
                        'Flat line areas', 'Decline Areas', 'Decline %']
    for col in required_columns:
        if col not in processed_df.columns:
            processed_df[col] = None

    # Add duration column
    processed_df['Total Duration'] = max(df['Video Start'].apply(time_to_minutes))

    
    return processed_df

def create_multiline_chart(dfs, x_column, y_column, title, colors, dnf=False):
    fig = go.Figure()
    for i, (video_title, df) in enumerate(dfs.items()):
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df[y_column],
            mode='lines',
            name=video_title,
            line=dict(color=colors[i % len(colors)])
        ))
        if dnf:
            decline_x = df[df['Decline Areas'] == 1][x_column]
            decline_y = df[df['Decline Areas'] == 1][y_column]
            fig.add_trace(go.Scatter(
                x=decline_x,
                y=decline_y,
                mode='markers',
                marker=dict(color='red', size=6),
                showlegend=False
            ))
            flats_x = df[df['Flat line areas'] == 1][x_column]
            flats_y = df[df['Flat line areas'] == 1][y_column]
            fig.add_trace(go.Scatter(
                x=flats_x,
                y=flats_y,
                mode='markers',
                marker=dict(color='green', size=6),
                showlegend=False
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


if file:
    # Read all sheets into a dictionary of dataframes
    sheets = pd.read_excel(file, sheet_name=None)

    # Process each sheet
    dfs = {sheet_name: process_data(df, sheet_name) for sheet_name, df in sheets.items()}

    # Create a dictionary to store return viewers count for filtering
    return_viewers_counts = {sheet_name: df['Return viewers'].sum() for sheet_name, df in dfs.items()}
    # Create a dictionary to store duration for filtering
    durations = {sheet_name: df['Total Duration'].max() for sheet_name, df in dfs.items()}

    # Get video titles from the sheet names
    video_titles = list(dfs.keys())

    # Filter by Return viewers
    all_return_viewers = [return_viewers_counts.get(title, 0) for title in video_titles]
    min_return_viewers = int(min(all_return_viewers, default=0))
    max_return_viewers = int(max(all_return_viewers, default=1000))  # Default to 1000 if max is 0

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

    # Filter by Duration
    all_durations = [durations.get(title, 0) for title in video_titles]
    min_duration = int(min(all_durations, default=0))
    max_duration = int(max(all_durations, default=1000))  # Default to 1000 if max is 0

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
           selected_min_duration <= durations.get(title, 0) <= selected_max_duration
    ]

    # Video selection filter
    selected_video = st.sidebar.selectbox('Select Video:', options=filtered_video_titles)

    # Filter for Viewer Type
    viewer_type = st.sidebar.selectbox('Select Viewer Type:', options=['All', 'return', 'new'])

    # Show or hide decline and flat areas
    dnf = st.sidebar.checkbox("Show Decline and Flats")

    # Get the dataframe for the selected video
    df = dfs[selected_video]

    # Filter by Viewer Type if not 'All'
    if viewer_type != 'All':
        df = df[df['ViewerType'] == viewer_type]

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
        
    fig_all_videos = create_multiline_chart(filtered_dfs, 'Video position (%)', 'Retention Start (%)', 'User Retention Chart for All Videos', colors, dnf)

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
    st.subheader('User Retention Chart for All Videos')
    st.plotly_chart(fig_all_videos, use_container_width=True)

    # Show the multiline chart for all videos
    st.subheader('User Retention Chart Per Second Retention Rate for All Videos')

    # Display processed data in a scrollable table at the bottom
    st.subheader('Processed Data')
    st.dataframe(df, height=200)

else:
    st.info("Upload the excel sheet first to continue")
