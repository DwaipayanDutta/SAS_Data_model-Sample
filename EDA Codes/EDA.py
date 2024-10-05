import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# Load and execute the Python script from GitHub
url = 'https://raw.githubusercontent.com/DwaipayanDutta/SAS_Data_model-Sample/main/Data/Updated_Data.py'
exec(requests.get(url).text)

# Assuming the DataFrame is named `df` after running the script
print(df.head())

# Set the aesthetic style of the plots
sns.set(style='whitegrid')

# Create a PowerPoint presentation object
ppt = Presentation()

# Function to format title text in Calibri 16pt
def set_title_format(title_shape):
    title_shape.text_frame.paragraphs[0].font.name = 'Calibri'
    title_shape.text_frame.paragraphs[0].font.size = Pt(16)
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 0, 0)  # Black color

# Function to save plots to PowerPoint
def save_plot_to_ppt(plot_func, slide_title):
    plt.figure(figsize=(10, 5))
    plot_func()
    plt.title(slide_title)
    plt.tight_layout()
    
    # Save the plot to a temporary file
    plt.savefig('temp_plot.png')
    plt.close()
    
    # Add a slide to the presentation
    slide = ppt.slides.add_slide(ppt.slide_layouts[5])  # Title Only layout
    title = slide.shapes.title
    title.text = slide_title.title()  # Convert to title case
    
    set_title_format(title)  # Set title formatting
    
    # Add image to slide
    left = Inches(1)
    top = Inches(1.5)
    slide.shapes.add_picture('temp_plot.png', left, top, width=Inches(8))

# Continuous Variables Analysis
continuous_vars = df.select_dtypes(include=['float64', 'int64']).columns

# Violin Plots for Continuous Variables
for var in continuous_vars:
    save_plot_to_ppt(lambda: sns.violinplot(x='LI_FLAG', y=var, data=df, palette='viridis'), f'Violin Plot of {var} by LI_FLAG')

# Correlation Analysis
correlation_matrix = df[continuous_vars].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('temp_correlation_heatmap.png')
plt.close()

# Add heatmap to PowerPoint
slide = ppt.slides.add_slide(ppt.slide_layouts[5])  # Title Only layout
title = slide.shapes.title
title.text = 'Correlation Heatmap'
set_title_format(title)  # Set title formatting

slide.shapes.add_picture('temp_correlation_heatmap.png', left=Inches(1), top=Inches(1.5), width=Inches(8))

# Categorical Variables Analysis
categorical_vars = df.select_dtypes(include=['object']).columns

for var in categorical_vars:
    save_plot_to_ppt(lambda: sns.countplot(x=df[var], palette='Set2'), f'Count of {var}')

# Bivariate Analysis: Continuous vs. Categorical with Box Plots
for var in continuous_vars:
    save_plot_to_ppt(lambda: sns.boxplot(x='LI_FLAG', y=var, data=df, palette='pastel'), f'Box Plot of {var} by LI_FLAG')

# Outlier Analysis using Box Plots for Continuous Variables
for var in continuous_vars:
    save_plot_to_ppt(lambda: sns.boxplot(y=var, data=df, palette='pastel'), f'Outlier Analysis for {var}')

# KDE Plots for Continuous Variables
for var in continuous_vars:
    save_plot_to_ppt(lambda: sns.kdeplot(data=df[var], shade=True), f'KDE Plot of {var}')

# Categorical vs. Categorical with Stacked Bar Plots (Crosstab)
for var in categorical_vars:
    if var != 'LI_FLAG':
        crosstab = pd.crosstab(df[var], df['LI_FLAG'])
        crosstab.plot(kind='bar', stacked=True, figsize=(10, 5), color=sns.color_palette("Paired"))
        plt.title(f'Stacked Bar Plot of {var} by LI_FLAG')
        plt.tight_layout()
        plt.savefig('temp_stacked_bar.png')
        plt.close()

        # Add stacked bar plot to PowerPoint
        slide = ppt.slides.add_slide(ppt.slide_layouts[5])  # Title Only layout
        title = slide.shapes.title
        title.text = f'Stacked Bar Plot of {var} by LI_FLAG'
        set_title_format(title)  # Set title formatting
        
        slide.shapes.add_picture('temp_stacked_bar.png', left=Inches(1), top=Inches(1.5), width=Inches(8))

# Save the PowerPoint presentation and print the path
ppt_file_path = 'Data_Analysis_Presentation.pptx'
ppt.save(ppt_file_path)
print(f"Presentation saved at: {ppt_file_path}")