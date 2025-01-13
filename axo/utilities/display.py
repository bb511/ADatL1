import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import math
import io
import base64
from weasyprint import HTML, CSS

def generate_axolotl_html_report(config, dict_axo, histogram_dict, threshold_dict, history_dict, output_file, pdf_file=None):
    plt.style.use('default')
    def dict_to_html_table(config_dict):
        html = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
        for key, value in config_dict.items():
            if isinstance(value, dict):
                html += f"<tr><td><strong>{key}</strong></td><td>{dict_to_html_table(value)}</td></tr>"
            else:
                html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    sorted_dict_axo = dict(sorted(dict_axo.items(), key=lambda item: float(item[0])))
    # Generate the HTML content for each section
    data_config_html = dict_to_html_table(config['data_config'])
    
    data_config_rem = config['data_config'].copy()
    data_config_rem.pop("Read_configs")
    
    data_config_html_read_bkg_html = dict_to_html_table(config['data_config']["Read_configs"]["BACKGROUND"])
    data_config_html_read_sig_html = dict_to_html_table(config['data_config']["Read_configs"]["SIGNAL"])
    data_config_html_read_rem_html = dict_to_html_table(data_config_rem)
    
    train_config_html = dict_to_html_table(config['train'])
    determinism_config_html = dict_to_html_table(config['determinism'])
    model_config_html = dict_to_html_table(config['model'])
    callback_config_html = dict_to_html_table(config['callback'])
    threshold_config_html = dict_to_html_table(config['threshold'])
    store_config_html = dict_to_html_table(config['store'])

    # Start building the full HTML content
    html_output = f"""
    <html>
    <head>
        <title>AXOLOTL1 Configuration</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            table {{
                width: 100%;
                margin-bottom: 20px;
                border: 1px solid #ccc;
                border-collapse: collapse;
                page-break-inside: auto;
            }}
            tr {{
                page-break-inside: avoid;
                page-break-after: auto;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            td {{
                vertical-align: top;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
            @page {{
                size: A4 landscape;
                margin: 1cm;
            }}
            .page-break {{
                page-break-before: always;
            }}
        </style>
    </head>
    <body>

    
    
    <div class="page-break">
    <h1>Configuration for AXOLOTL1</h1>
    <h2>Read Configuration - ZEROBIAS</h2>
    {data_config_html_read_bkg_html}
    </div>
    
    <div class="page-break">
    <h2>Read Configuration - MonteCarlo</h2>
    {data_config_html_read_sig_html}
    </div>

    <div class="page-break">
    <h2>Preprocessing Configs</h2>
    {data_config_html_read_rem_html}
    </div>

    
    <div class="page-break">
        <h2>Training Configuration</h2>
        {train_config_html}
    </div>

    <div class="page-break">
        <h2>Determinism Configuration</h2>
        {determinism_config_html}
    </div>

    <div class="page-break">
        <h2>Model Configuration</h2>
        {model_config_html}
    </div>

    <div class="page-break">
        <h2>Callback Configuration</h2>
        {callback_config_html}
    </div>

    <div class="page-break">
        <h2>Threshold Configuration</h2>
        {threshold_config_html}
    </div>

    <div class="page-break">
        <h2>Store Configuration</h2>
        {store_config_html}
    </div>
    """

    # Number of columns for the grid layout for the tables
    num_cols = int(math.ceil(math.sqrt(len(config))))  # Calculate the square root and round up to get a square grid

    # Loop through each threshold and its corresponding DataFrame in the sorted dictionary
    for threshold, df in sorted_dict_axo.items():
        html_output += f"""
        <div class="page-break">
            <h1>AXO Score Table for Threshold: {threshold} kHz</h1>
            {df.to_html(index=False, escape=False)}
        </div>
        """

    # Now let's generate and add the first figure with histograms

    # Number of columns and rows for the grid
    num_cols = 6
    num_rows = 6

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(36, 18))

    # Flatten axes array to easily iterate over it if it's a 2D array
    axes = axes.flatten()

    # Plot the background histogram in the first subplot
    axes[0].set_title("ZeroBias (Background)")
    hep.histplot(histogram_dict['background'], ax=axes[0], color='blue')
    axes[0].set_yscale("log")

    # Add vertical lines for each threshold
    for thres, value in threshold_dict.items():
        axes[0].axvline(x=value, linestyle='--', linewidth=2, label=f"{thres} kHz")

    axes[0].grid()
    axes[0].legend()

    # Plot histograms for each signal in the remaining subplots
    plot_index = 1  # Start with the second subplot (index 1)
    for signal_name, hist_data in histogram_dict.items():
        if signal_name == "background":
            continue  # Skip the background as it's already plotted

        ax = axes[plot_index]  # Use the next subplot
        ax.set_title(signal_name)
        hep.histplot(hist_data, ax=ax)
        ax.set_yscale("log")

        # Add vertical lines for each threshold
        for thres, value in threshold_dict.items():
            ax.axvline(x=value, linestyle='--', linewidth=2, label=f"{thres} kHz")
        
        ax.grid()
        ax.legend()
        
        plot_index += 1  # Move to the next subplot

    # Hide any unused subplots
    for j in range(plot_index, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Save the figure to a bytes buffer instead of a file
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the image to a base64 string
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    # Embed the image in the HTML
    html_output += f"""
    <div class="page-break">
        <h1>Signal Histograms</h1>
        <img src="data:image/png;base64,{img_str}" alt="Signal Histograms">
    </div>
    """

    # Now let's generate and add the second figure (AXO scores vs. thresholds)

    # Sort the thresholds in ascending order
    sorted_thresholds = sorted(dict_axo.keys(), key=lambda x: float(x))

    ##################################################################################################
    # Extract all unique signal names from the first DataFrame (assuming all DataFrames have the same signal names)
    signal_names = dict_axo[sorted_thresholds[0]]['Signal Name'].tolist()

    # Determine the number of signals
    num_signals = len(signal_names)

    # Determine the number of rows and columns for a square grid
    num_cols = num_rows = math.ceil(math.sqrt(num_signals))

    # Create subplots in a square grid layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 20))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Plot each signal in a separate subplot
    for i, signal_name in enumerate(signal_names):
        ax = axes[i]

        # Prepare x and y data
        x_data = [float(threshold) for threshold in sorted_thresholds]
        y_data = [dict_axo[threshold].loc[dict_axo[threshold]['Signal Name'] == signal_name, 'AXO SCORE'].values[0] for threshold in sorted_thresholds]
        
        # Plot the data
        ax.plot(x_data, y_data, marker='o', label=signal_name)
        ax.set_title(signal_name)
        ax.set_xlabel('Threshold (kHz)')
        ax.set_ylabel('AXO SCORE')
        ax.set_yscale('log')  # Set y-axis to logarithmic scale
        ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Set the overall title for the figure
    fig.suptitle('AXO for different Signals', fontsize=20)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust to fit the suptitle

    # Save the figure to a bytes buffer instead of a file
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the image to a base64 string
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    # Embed the image in the HTML
    html_output += f"""
    <div class="page-break">
        <h1>AXO for different Signals</h1>
        <img src="data:image/png;base64,{img_str}" alt="AXO for different Signals">
    </div>
    """
    


    ###################################################################################################
    # Apply the CMS style
    plt.style.use(hep.style.CMS)

    # Determine the number of rows and columns for a square grid based on the history_dict size
    num_history_items = len(history_dict)
    num_cols = num_rows = math.ceil(math.sqrt(num_history_items))

    # Create subplots in a square grid layout with a larger figure size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 18))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Plot each item in history_dict with enhanced legibility
    for i, (key, values) in enumerate(history_dict.items()):
        ax = axes[i]

        # Plot the values with epochs as the x-axis
        ax.plot(range(1, len(values) + 1), values, label=key, linewidth=2)  # Increase line width
        ax.set_title(key, fontsize=20, pad=15)  # Increase title font size and add padding
        ax.set_xlabel('Epoch', fontsize=18, labelpad=10)  # Increase x-axis label font size and add padding
        # ax.set_ylabel(key, fontsize=18, labelpad=10)  # Increase y-axis label font size and add padding
        ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
        ax.grid(True, linestyle='--', linewidth=0.7)  # Enhance grid line visibility

        # Center the axis labels
        ax.xaxis.set_label_coords(0.5, -0.1)
        # ax.yaxis.set_label_coords(-0.1, 0.7)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle('History for AXO Training', fontsize=24)
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    plt.style.use('default')



    ########################################################################################################
    # Save the figure to a bytes buffer instead of a file
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the image to a base64 string
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    # Embed the image in the HTML
    html_output += f"""
    <div style="page-break-inside: avoid;">
        <img src="data:image/png;base64,{img_str}" alt="Training History" style="max-width:100%; max-height:80vh;">
    </div>
    """

    # Close the HTML body and HTML tag
    html_output += "</body></html>"

    # Write the HTML content to a file
    if output_file is not None:
        with open(output_file, "w") as file:
            file.write(html_output)
        print(f"HTML file generated: {output_file}")
    
    if pdf_file is not None:
        HTML(string=html_output).write_pdf(pdf_file, stylesheets=[CSS(string='''
        @page { 
            size: A4 landscape; 
            margin: 1cm;
            @bottom-left {
                content: counter(page);
            } 
        } 
        body {
            font-size: 10pt;
        }
        h1, h2, h3 {
            font-size: 12pt;
        }
        table {
            font-size: 9pt;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        ''')])
        print(f"PDF file generated: {pdf_file}")

