"""
Visualization utilities for Streamlit app.
Creates interactive charts using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px


def create_probability_chart(probabilities):
    """
    Create horizontal bar chart for emotion probabilities.
    
    Args:
        probabilities: Dictionary of emotion: probability
        
    Returns:
        Plotly figure
    """
    # Sort by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    emotions = [e.replace('_', ' ').title() for e, _ in sorted_probs]
    probs = [p * 100 for _, p in sorted_probs]
    
    # Create figure
    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation='h',
        marker=dict(
            color=probs,
            colorscale='RdYlGn',
            showscale=False
        ),
        text=[f'{p:.1f}%' for p in probs],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Emotion Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_confidence_gauge(confidence):
    """
    Create gauge chart for confidence score.
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 75], 'color': '#ffffcc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_model_contribution_chart(modalities, weights):
    """
    Create pie chart showing model contributions.
    
    Args:
        modalities: List of modality names
        weights: Dictionary of modality: weight
        
    Returns:
        Plotly figure
    """
    # Map modality names to display names
    display_names = {
        'audio': '🎵 Audio',
        'facial': '📸 Facial',
        'text': '💬 Text'
    }
    
    labels = [display_names.get(m, m) for m in modalities]
    values = [weights[m] * 100 for m in modalities]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    )])
    
    fig.update_layout(
        title="Model Contributions",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )
    
    return fig


def format_prediction_result(result):
    """
    Format prediction result for display.
    
    Args:
        result: Prediction dictionary
        
    Returns:
        Formatted string
    """
    if 'error' in result:
        return f"Error: {result['error']}"
    
    emotion = result['prediction'].replace('_', ' ').title()
    confidence = result['confidence'] * 100
    
    return f"**{emotion}** ({confidence:.1f}% confident)"
