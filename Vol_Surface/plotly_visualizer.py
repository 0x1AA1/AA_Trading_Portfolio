"""
Interactive Plotly Visualizations for Volatility Surfaces

Based on 2024-2025 research recommendations for modern volatility surface visualization
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.interpolate import griddata


class VolatilitySurfaceVisualizer:
    """
    Create interactive 3D volatility surface visualizations using Plotly
    """

    def __init__(self, theme='plotly_dark'):
        """
        Args:
            theme: Plotly theme ('plotly', 'plotly_dark', 'plotly_white')
        """
        self.theme = theme

    def plot_3d_surface(self, strikes, maturities, iv_surface, spot_price=None,
                       title="Implied Volatility Surface", save_html=None):
        """
        Create interactive 3D volatility surface

        Args:
            strikes: Array of strike prices (K,)
            maturities: Array of maturities in years (T,)
            iv_surface: 2D array of implied volatilities (K, T)
            spot_price: Current spot price (optional, for reference line)
            title: Plot title
            save_html: Path to save HTML file (optional)

        Returns:
            plotly Figure object
        """
        # Create meshgrid for plotting
        X, Y = np.meshgrid(maturities, strikes)

        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=iv_surface * 100,  # Convert to percentage
            colorscale='Viridis',
            colorbar=dict(title='IV (%)', x=1.1),
            hovertemplate='<b>Maturity:</b> %{x:.2f} years<br>' +
                         '<b>Strike:</b> $%{y:.2f}<br>' +
                         '<b>IV:</b> %{z:.2f}%<br>' +
                         '<extra></extra>',
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
            contours=dict(
                z=dict(show=True, usecolormap=True, width=2, highlightwidth=5)
            )
        )])

        # Add spot price reference line if provided
        if spot_price is not None:
            spot_line = np.full(len(maturities), spot_price)
            z_values = np.array([iv_surface[np.argmin(np.abs(strikes - spot_price)), i]
                                for i in range(len(maturities))]) * 100

            fig.add_trace(go.Scatter3d(
                x=maturities,
                y=spot_line,
                z=z_values,
                mode='lines',
                line=dict(color='red', width=6),
                name='ATM Strike',
                hovertemplate='ATM: $%{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            scene=dict(
                xaxis=dict(title='Time to Maturity (years)', backgroundcolor='rgb(230, 230,230)'),
                yaxis=dict(title='Strike Price ($)', backgroundcolor='rgb(230, 230,230)'),
                zaxis=dict(title='Implied Volatility (%)', backgroundcolor='rgb(230, 230,230)'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            template=self.theme,
            width=1200,
            height=800,
            showlegend=True
        )

        if save_html:
            fig.write_html(save_html)

        return fig

    def plot_smile_comparison(self, strikes, ivs_list, labels, spot_price=None,
                            title="Volatility Smile Comparison"):
        """
        Compare multiple volatility smiles

        Args:
            strikes: Array of strike prices
            ivs_list: List of IV arrays to compare
            labels: List of labels for each smile
            spot_price: Current spot price (optional)
            title: Plot title

        Returns:
            plotly Figure
        """
        fig = go.Figure()

        colors = px.colors.qualitative.Set2

        for i, (ivs, label) in enumerate(zip(ivs_list, labels)):
            fig.add_trace(go.Scatter(
                x=strikes,
                y=ivs * 100,
                mode='lines+markers',
                name=label,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=8),
                hovertemplate='<b>Strike:</b> $%{x:.2f}<br>' +
                             '<b>IV:</b> %{y:.2f}%<br>' +
                             f'<b>Source:</b> {label}<extra></extra>'
            ))

        # Add ATM line if spot provided
        if spot_price is not None:
            fig.add_vline(x=spot_price, line_dash="dash", line_color="red",
                         annotation_text="ATM", annotation_position="top")

        fig.update_layout(
            title=title,
            xaxis_title='Strike Price ($)',
            yaxis_title='Implied Volatility (%)',
            template=self.theme,
            width=1200,
            height=600,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
        )

        return fig

    def plot_term_structure(self, maturities, atm_ivs, title="Volatility Term Structure"):
        """
        Plot ATM volatility term structure

        Args:
            maturities: Array of maturities (years)
            atm_ivs: Array of ATM implied volatilities
            title: Plot title

        Returns:
            plotly Figure
        """
        # Convert to days for better readability
        days = maturities * 365

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=days,
            y=atm_ivs * 100,
            mode='lines+markers',
            name='ATM IV',
            line=dict(width=3, color='#636EFA'),
            marker=dict(size=10, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(99, 110, 250, 0.2)',
            hovertemplate='<b>Days:</b> %{x:.0f}<br>' +
                         '<b>ATM IV:</b> %{y:.2f}%<extra></extra>'
        ))

        # Add trend line
        z = np.polyfit(days, atm_ivs * 100, 1)
        p = np.poly1d(z)
        trend_line = p(days)

        fig.add_trace(go.Scatter(
            x=days,
            y=trend_line,
            mode='lines',
            name=f'Trend (slope: {z[0]:.4f})',
            line=dict(width=2, dash='dash', color='red'),
            hovertemplate='<b>Trend:</b> %{y:.2f}%<extra></extra>'
        ))

        # Determine structure shape
        shape = 'CONTANGO' if z[0] > 0.01 else 'BACKWARDATION' if z[0] < -0.01 else 'FLAT'

        fig.update_layout(
            title=f"{title} - {shape}",
            xaxis_title='Days to Expiration',
            yaxis_title='ATM Implied Volatility (%)',
            template=self.theme,
            width=1200,
            height=600,
            showlegend=True
        )

        return fig

    def plot_comprehensive_dashboard(self, strikes, maturities, iv_surface, spot_price,
                                    greeks_data=None, title="Volatility Surface Dashboard"):
        """
        Create comprehensive dashboard with multiple views

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            iv_surface: 2D IV array (strikes x maturities)
            spot_price: Current spot price
            greeks_data: Optional dict with Greeks surfaces
            title: Dashboard title

        Returns:
            plotly Figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D Volatility Surface', 'Volatility Smile (Near Term)',
                          'Term Structure (ATM)', 'Heatmap'),
            specs=[[{'type': 'surface', 'rowspan': 2}, {'type': 'xy'}],
                   [None, {'type': 'xy'}]],
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )

        # 1. 3D Surface (spans 2 rows)
        X, Y = np.meshgrid(maturities, strikes)
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=iv_surface * 100,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=-0.15, len=0.9, title='IV (%)'),
                hovertemplate='T: %{x:.2f}y, K: $%{y:.0f}<br>IV: %{z:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Volatility Smile (first maturity)
        near_term_iv = iv_surface[:, 0] * 100
        moneyness = strikes / spot_price

        fig.add_trace(
            go.Scatter(
                x=moneyness,
                y=near_term_iv,
                mode='lines+markers',
                name=f'T={maturities[0]:.2f}y',
                line=dict(width=2, color='#636EFA'),
                marker=dict(size=6)
            ),
            row=1, col=2
        )

        fig.add_vline(x=1.0, line_dash="dash", line_color="red", row=1, col=2)

        # 3. Term Structure (ATM)
        atm_idx = np.argmin(np.abs(strikes - spot_price))
        atm_ivs = iv_surface[atm_idx, :] * 100

        fig.add_trace(
            go.Scatter(
                x=maturities * 365,
                y=atm_ivs,
                mode='lines+markers',
                name='ATM Term Structure',
                line=dict(width=2, color='#EF553B'),
                marker=dict(size=8, symbol='diamond'),
                fill='tozeroy',
                fillcolor='rgba(239, 85, 59, 0.2)'
            ),
            row=2, col=2
        )

        # Update axes
        fig.update_xaxes(title_text="Moneyness (K/S)", row=1, col=2)
        fig.update_yaxes(title_text="IV (%)", row=1, col=2)

        fig.update_xaxes(title_text="Days to Expiration", row=2, col=2)
        fig.update_yaxes(title_text="ATM IV (%)", row=2, col=2)

        fig.update_scenes(
            xaxis_title='Maturity (years)',
            yaxis_title='Strike ($)',
            zaxis_title='IV (%)',
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.2)),
            row=1, col=1
        )

        fig.update_layout(
            title_text=title,
            template=self.theme,
            height=900,
            width=1600,
            showlegend=True
        )

        return fig

    def plot_arbitrage_violations(self, strikes, maturities, iv_surface,
                                 calendar_violations=None, butterfly_violations=None,
                                 title="Arbitrage Violation Heatmap"):
        """
        Visualize arbitrage violations in the surface

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            iv_surface: 2D IV array
            calendar_violations: 2D array of calendar spread violations
            butterfly_violations: 2D array of butterfly violations
            title: Plot title

        Returns:
            plotly Figure
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('IV Surface', 'Calendar Violations', 'Butterfly Violations'),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
        )

        # 1. Original surface
        fig.add_trace(
            go.Heatmap(
                x=maturities,
                y=strikes,
                z=iv_surface * 100,
                colorscale='Viridis',
                colorbar=dict(x=0.31, title='IV (%)'),
                hovertemplate='T: %{x:.2f}y<br>K: $%{y:.0f}<br>IV: %{z:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Calendar violations
        if calendar_violations is not None:
            fig.add_trace(
                go.Heatmap(
                    x=maturities,
                    y=strikes,
                    z=calendar_violations,
                    colorscale='Reds',
                    colorbar=dict(x=0.65, title='Violation'),
                    hovertemplate='Violation: %{z:.4f}<extra></extra>'
                ),
                row=1, col=2
            )

        # 3. Butterfly violations
        if butterfly_violations is not None:
            fig.add_trace(
                go.Heatmap(
                    x=maturities,
                    y=strikes,
                    z=butterfly_violations,
                    colorscale='Reds',
                    colorbar=dict(x=1.0, title='Violation'),
                    hovertemplate='Violation: %{z:.4f}<extra></extra>'
                ),
                row=1, col=3
            )

        fig.update_xaxes(title_text="Maturity (years)")
        fig.update_yaxes(title_text="Strike ($)")

        fig.update_layout(
            title_text=title,
            template=self.theme,
            height=600,
            width=1800
        )

        return fig

    def plot_pnl_distribution(self, spot_range, pnl_expiration, pnl_current=None,
                            breakeven_points=None, title="Straddle P&L Distribution"):
        """
        Plot P&L distribution for options strategies

        Args:
            spot_range: Array of spot prices
            pnl_expiration: P&L at expiration
            pnl_current: Current P&L (optional)
            breakeven_points: Tuple of (lower, upper) breakeven prices
            title: Plot title

        Returns:
            plotly Figure
        """
        fig = go.Figure()

        # P&L at expiration
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=pnl_expiration,
            mode='lines',
            name='P&L at Expiration',
            line=dict(width=3, color='#636EFA'),
            fill='tozeroy',
            fillcolor='rgba(99, 110, 250, 0.2)',
            hovertemplate='Spot: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
        ))

        # Current P&L if provided
        if pnl_current is not None:
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=pnl_current,
                mode='lines',
                name='Current P&L',
                line=dict(width=2, dash='dash', color='#EF553B'),
                hovertemplate='Spot: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
            ))

        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        # Breakeven points
        if breakeven_points is not None:
            lower_be, upper_be = breakeven_points
            fig.add_vline(x=lower_be, line_dash="dash", line_color="green",
                         annotation_text=f"Lower BE: ${lower_be:.2f}",
                         annotation_position="top left")
            fig.add_vline(x=upper_be, line_dash="dash", line_color="green",
                         annotation_text=f"Upper BE: ${upper_be:.2f}",
                         annotation_position="top right")

        fig.update_layout(
            title=title,
            xaxis_title='Underlying Price ($)',
            yaxis_title='Profit/Loss ($)',
            template=self.theme,
            width=1200,
            height=600,
            hovermode='x unified'
        )

        return fig

    def plot_greeks_surface(self, strikes, maturities, greeks_dict,
                          greek_name='delta', title=None):
        """
        Plot 3D surface of a specific Greek

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            greeks_dict: Dictionary with Greek values (K x T array)
            greek_name: Name of Greek to plot
            title: Plot title (auto-generated if None)

        Returns:
            plotly Figure
        """
        if title is None:
            title = f"{greek_name.capitalize()} Surface"

        greek_values = greeks_dict[greek_name]
        X, Y = np.meshgrid(maturities, strikes)

        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=greek_values,
            colorscale='RdBu',
            colorbar=dict(title=greek_name.capitalize()),
            hovertemplate='T: %{x:.2f}y<br>K: $%{y:.0f}<br>' +
                         f'{greek_name}: %{{z:.4f}}<extra></extra>'
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Maturity (years)',
                yaxis_title='Strike ($)',
                zaxis_title=greek_name.capitalize(),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            template=self.theme,
            width=1200,
            height=800
        )

        return fig
