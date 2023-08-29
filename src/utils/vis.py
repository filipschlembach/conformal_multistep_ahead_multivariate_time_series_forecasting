import numpy as np
import plotly.graph_objects as go


def show_results(x, y, y_hat: dict, wl, hl, s, x_all: np.ndarray = None):
    # todo: documentation

    # visualize with slider: https://plotly.com/python/sliders/
    fig = go.Figure()

    # Add traces, one for each slider step
    for w_i in range(0, x.shape[0]):  # W_0
        x_range = np.arange(w_i * s, w_i * s + wl)
        y_range = np.arange(w_i * s + wl, w_i * s + wl + hl)
        for a in y_hat.keys():
            if a == 'model':
                continue
            fig.add_trace(go.Scatter(
                visible=False,
                x=np.concatenate([y_range, y_range[::-1]]),
                y=np.concatenate([y_hat[a][w_i, :, 0, 0], y_hat[a].numpy()[w_i, ::-1, 0, 1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.1)',
                line_color='rgba(255,255,255,0)',
                name=f'{(1 - a) * 100} % interval (w {w_i})'))
        fig.add_trace(go.Scatter(
            visible=False,
            line=dict(color="blue", width=2),
            name=f'x {w_i}',
            x=x_range,
            y=x[w_i, :, 0]))
        fig.add_trace(go.Scatter(
            visible=False,
            line=dict(color="red", width=2),
            name=f'y {w_i}',
            x=y_range,
            y=y[w_i, :, 0]))
        fig.add_trace(go.Scatter(
            visible=False,
            line=dict(color="green", width=2),
            name=f'ŷ {w_i}',
            x=y_range,
            y=y_hat['model'][w_i, :, 0]))

    trances_per_step = 2 + len(y_hat)
    # Make 0th trace visible
    for i in range(trances_per_step):
        fig.data[i].visible = True

    # add all time steps of the time series
    if x_all is not None:
        x_range = np.arange(0, len(x_all))
        fig.add_trace(go.Scatter(
            line=dict(color="gray", width=1),
            name='data',
            x=x_range,
            y=x_all
        ))

    # Create and add slider
    steps = []
    for i in range(0, len(fig.data) // trances_per_step):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to window: " + str(i)}],  # layout attribute
        )
        for j in range(trances_per_step):
            step["args"][0]["visible"][trances_per_step * i + j] = True  # Toggle i'th trace to "visible"
        if x_all is not None:
            step["args"][0]["visible"][-1] = True  # Toggle complete time series to visible
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Window: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.show()
