# 🚀 Chain-of-Influence Demos

This directory contains interactive demonstrations of the Chain-of-Influence (CoI) model and its interpretability features.

## 📓 Google Colab Interactive Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/demos/colab/demo.ipynb)

### `colab/demo.ipynb` - Complete Interactive Experience

A comprehensive Google Colab notebook that provides:

#### 🎯 **Key Features**

- **📁 File Upload Interface**: Drag-and-drop support for CSV, Excel, JSON influence matrices
- **🔍 Data Processing**: Automatic parsing with robust error handling and data validation
- **🌐 Interactive Network Visualization**: 
  - Physics-based layout with adjustable parameters
  - Color-coded influence relationships (blue: positive, red: negative)
  - Node sizing based on feature importance
  - Edge thickness proportional to influence strength
- **🤖 AI-Powered Analysis**: OpenAI GPT-4 integration for intelligent pattern recognition
- **📊 Professional Dashboard**: Six-tab interface with cloud-style design

#### 📋 **Dashboard Tabs**

1. **Model Performance** - Key metrics (AUROC, F1, Precision, Recall, Accuracy)
2. **Key Timestamps** - Critical time points in disease progression
3. **Key Features** - Feature importance rankings and analysis
4. **Cross Analysis** - Interactive network visualization with physics controls
5. **LLM Summary** - AI-generated insights and pattern recognition
6. **Token Usage** - API usage tracking and cost monitoring

#### 🛠️ **Technical Capabilities**

- **Smart Data Handling**: 
  - Auto-detection of file formats
  - Graceful handling of NaN values and temporal constraints
  - Configurable influence thresholds for network filtering
- **Advanced Visualizations**:
  - Heatmaps with diverging colormaps
  - Interactive network graphs with real-time physics
  - Professional styling with responsive design
- **AI Integration**:
  - Pattern recognition in temporal-feature relationships
  - Clinical insight generation
  - Automated summarization of influence chains

## 🚀 **Quick Start**

### **One-Click Launch - No Installation Required!**

1. **Click the "Open in Colab" badge above** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/demos/colab/demo.ipynb)
2. **Upload your influence matrix** file (CSV/Excel format) or use our sample data
3. **Set your OpenAI API key** in Colab secrets as `OPENAI_API_KEY` (optional - for AI analysis)
4. **Run all cells** to generate the interactive dashboard

**That's it!** Your professional dashboard will appear with interactive visualizations and AI insights.

## 📊 **Expected Input Format**

The demo expects influence matrix files with:

- **Rows**: Source features with temporal labels (e.g., `eGFR_t-0`, `CHF_t-1`)
- **Columns**: Target features with temporal labels
- **Values**: Influence scores (positive/negative floats, NaN for invalid relationships)
- **Format**: CSV, Excel (.xlsx/.xls), or JSON

### Example CSV Structure:
```
,eGFR_t-0,CHF_t-0,Hemoglobin_t-0,eGFR_t-1,CHF_t-1
eGFR_t-0,NaN,0.0234,0.0156,0.0891,0.0234
CHF_t-0,NaN,NaN,0.0167,0.0234,0.0445
Hemoglobin_t-0,NaN,NaN,NaN,0.0123,0.0089
```

## 🎨 **Visualization Gallery**

The demo generates multiple types of visualizations:

### 🔗 **Network Visualizations**
- **Interactive Network**: Physics-based layout with hover tooltips
- **Influence Heatmap**: Matrix view with diverging colormap
- **Feature Importance**: Ranked bar charts and temporal patterns

### 📈 **Analysis Outputs**
- **Top Influence Pairs**: Ranked list of strongest relationships
- **Temporal Patterns**: Time-series analysis of feature evolution
- **Clinical Insights**: AI-generated interpretations of influence chains

## 🔧 **Customization Options**

### Network Visualization
- `threshold`: Minimum influence magnitude to display (default: 0.001)
- `physics_settings`: Gravity, spring length, central gravity parameters
- `color_scheme`: Positive (blue) vs negative (red) influence colors

### AI Analysis
- `model`: OpenAI model selection (`gpt-4`, `gpt-4o`, `gpt-3.5-turbo`)
- `temperature`: Response creativity (0.0-1.0, default: 0.2)
- `analysis_depth`: Focused vs comprehensive analysis modes

### Dashboard Styling
- `theme`: Cloud-style tabs with green-framed containers
- `layout`: Responsive design with mobile compatibility
- `animations`: Smooth transitions and hover effects

## 🛡️ **Privacy & Security**

- **🔐 API Keys**: Stored securely in Colab secrets, never exposed in code
- **🖥️ Data Processing**: All computation happens in Google's secure cloud environment
- **📁 File Handling**: Temporary upload only during session, automatically cleaned up
- **🌐 Network Traffic**: Only optional API calls to OpenAI for AI analysis
- **🔒 Zero Installation**: No local setup required, runs entirely in browser

## 📞 **Support & Troubleshooting**

### 🚨 **Common Issues:**
1. **File Upload Fails**: Ensure CSV has proper headers and numeric values
2. **Network Too Dense**: Increase threshold parameter (try 0.01) to reduce edge count  
3. **API Errors**: Verify OpenAI API key is set correctly in Colab secrets (`OPENAI_API_KEY`)
4. **Empty Visualizations**: Use the provided sample data to test functionality first

### ⚡ **Performance Tips:**
- **Large Networks**: Use threshold > 0.01 for datasets with >1000 features
- **API Costs**: Monitor token usage in the dashboard's Token Usage tab
- **Browser Performance**: Refresh the page if network visualization becomes slow
- **Demo Testing**: Start with the sample CSV to verify everything works before uploading your data

## 🤝 **Contributing**

Found a bug or want to add features? 
1. Fork the repository
2. Create a feature branch
3. Test with sample data
4. Submit a pull request

## 📝 **Citation**

If you use this demo in your research, please cite:
```bibtex
@article{chain_of_influence_2024,
  title={Chain-of-Influence: Tracing Interdependencies Across Time and Features in Clinical Predictive Modeling},
  author={Your Name},
  journal={AAAI Conference on Artificial Intelligence},
  year={2024}
}
``` 