import plotly.graph_objects as go

def train_visualization(train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1):
    """
    Visualize Loss, Accuracy and F1
    """
    # Loss
    fig = go.Figure([
        go.Scatter(x=list(range(1, 301)), y=train_loss, name="Train Loss"),
        go.Scatter(x=list(range(1, 301)), y=val_loss, name="Val Loss")
                    ])
    fig.update_layout(title='Loss', xaxis_title='epoch')
    fig.show()
    fig.write_html("Viz/Loss.html")

    # Accuracy
    fig = go.Figure([
        go.Scatter(x=list(range(1, 301)), y=train_accuracy, name="Train Accuracy"),
        go.Scatter(x=list(range(1, 301)), y=val_accuracy, name="Val Accuracy")
                    ])
    fig.update_layout(title='Accuracy', xaxis_title='epoch')
    fig.show()
    fig.write_html("Viz/Accuracy.html")

    # F1
    fig = go.Figure([
        go.Scatter(x=list(range(1, 301)), y=train_f1, name="Train F1"),
        go.Scatter(x=list(range(1, 301)), y=val_f1, name="Val F1")
                    ])
    fig.update_layout(title='F1', xaxis_title='epoch')
    fig.show()
    fig.write_html("Viz/F1.html")