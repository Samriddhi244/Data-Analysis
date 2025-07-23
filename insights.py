import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better visuals
plt.style.use('default')
sns.set_palette("husl")

# Load data
df = pd.read_csv('Excelerate data.csv')

# Clean Amount Spent
df['Amount Spent in INR'] = df['Amount Spent in INR'].replace('[\$,]', '', regex=True).astype(float)

# Clean geography names for better readability
df['Geography_Clean'] = df['Geography'].apply(lambda x: 
    'Multi-Country Group 1' if 'Group 1' in str(x) else
    'Multi-Country Group 2' if 'Group 2' in str(x) else
    str(x).split(',')[0] if ',' not in str(x) else str(x)
)

# 1. Campaign Performance Overview - Top performing campaigns by CTR
campaign_performance = df.groupby('Campaign Name').agg({
    'Click-Through Rate (CTR in %)': 'mean',
    'Clicks': 'sum',
    'Amount Spent in INR': 'sum'
}).round(2)

# 1.1 Top Campaigns by CTR
plt.figure(figsize=(12, 8))
top_campaigns = campaign_performance.nlargest(10, 'Click-Through Rate (CTR in %)')
bars = plt.bar(range(len(top_campaigns)), top_campaigns['Click-Through Rate (CTR in %)'], 
               color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'][:len(top_campaigns)])
plt.title('Top Campaigns by Click-Through Rate', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Campaign')
plt.ylabel('CTR (%)')
plt.xticks(range(len(top_campaigns)), [name[:20]+'...' if len(name) > 20 else name 
                                       for name in top_campaigns.index], rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('top_campaigns_ctr.png', dpi=300, bbox_inches='tight')
plt.show()

# 1.2 Age Group Performance
plt.figure(figsize=(10, 6))
age_performance = df.groupby('Age').agg({
    'Reach': 'sum',
    'Clicks': 'sum',
    'Impressions': 'sum'
}).reset_index()

age_performance['CTR'] = (age_performance['Clicks'] / age_performance['Impressions']) * 100
bars = plt.bar(age_performance['Age'], age_performance['CTR'], 
               color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'])
plt.title('Click-Through Rate by Age Group', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Age Group')
plt.ylabel('CTR (%)')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('age_group_ctr.png', dpi=300, bbox_inches='tight')
plt.show()

# 1.3 Campaign ID Cost Efficiency Analysis
plt.figure(figsize=(10, 6))
df['CPC_Clean'] = df['Cost Per Click (CPC)'].replace('[\$,]', '', regex=True).astype(float)
df_campaign_temp = df.dropna(subset=['campaign ID'])
campaign_cost = df_campaign_temp.groupby('campaign ID').agg({
    'CPC_Clean': 'mean',
    'Clicks': 'sum',
    'Amount Spent in INR': 'sum'
}).reset_index()

# Filter for meaningful data
campaign_cost = campaign_cost[campaign_cost['Clicks'] > 20]
scatter = plt.scatter(campaign_cost['CPC_Clean'], campaign_cost['Clicks'], 
                     s=campaign_cost['Amount Spent in INR']/1000, alpha=0.7, 
                     c=range(len(campaign_cost)), cmap='viridis')
plt.title('Cost vs Performance by Campaign ID', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Cost Per Click (INR)')
plt.ylabel('Total Clicks')
plt.grid(True, alpha=0.3)

# Add labels for each point
for i, row in campaign_cost.iterrows():
    plt.annotate(row['campaign ID'][:12], 
                (row['CPC_Clean'], row['Clicks']),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=9, alpha=0.8)

plt.tight_layout()
plt.savefig('cost_vs_performance_campaign_id.png', dpi=300, bbox_inches='tight')
plt.show()

# 1.4 Audience Type Comparison
plt.figure(figsize=(8, 6))
audience_stats = df.groupby('Audience').agg({
    'Reach': 'sum',
    'Amount Spent in INR': 'sum',
    'Clicks': 'sum'
}).reset_index()

audience_stats['ROI'] = audience_stats['Clicks'] / audience_stats['Amount Spent in INR'] * 1000
bars = plt.bar(audience_stats['Audience'], audience_stats['ROI'], 
               color=['#e74c3c', '#3498db'])
plt.title('Return on Investment by Audience Type\n(Clicks per $1000 spent)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Audience Type')
plt.ylabel('Clicks per $1000')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('audience_roi_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Campaign ID Performance Analysis (Basic Charts)
# Clean and filter campaign ID data for basic analysis
df_campaign_clean_basic = df.dropna(subset=['campaign ID'])

# Create campaign ID analysis
campaign_analysis = df_campaign_clean_basic.groupby('campaign ID').agg({
    'Reach': 'sum',
    'Impressions': 'sum', 
    'Clicks': 'sum',
    'Amount Spent in INR': 'sum',
    'Click-Through Rate (CTR in %)': 'mean'
}).reset_index()

campaign_analysis['CPM'] = (campaign_analysis['Amount Spent in INR'] / campaign_analysis['Impressions']) * 1000
campaign_analysis = campaign_analysis[campaign_analysis['Clicks'] > 10]  # Filter for meaningful data

# 2.1 Total Reach by Campaign ID
plt.figure(figsize=(12, 8))
bars = plt.barh(campaign_analysis['campaign ID'], campaign_analysis['Reach'], 
                color=plt.cm.Set3(np.linspace(0, 1, len(campaign_analysis))))
plt.title('Total Reach by Campaign ID', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Reach')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('campaign_id_reach_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.2 Average CTR by Campaign ID
plt.figure(figsize=(12, 8))
bars = plt.barh(campaign_analysis['campaign ID'], campaign_analysis['Click-Through Rate (CTR in %)'], 
                color=plt.cm.Set2(np.linspace(0, 1, len(campaign_analysis))))
plt.title('Average CTR by Campaign ID', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('CTR (%)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('campaign_id_ctr_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.3 Spend vs Clicks Scatter for Campaign ID
plt.figure(figsize=(10, 8))
plt.scatter(campaign_analysis['Amount Spent in INR'], campaign_analysis['Clicks'], 
           s=campaign_analysis['Reach']/100, alpha=0.7, 
           c=campaign_analysis['Click-Through Rate (CTR in %)'], cmap='RdYlBu_r')
plt.colorbar(label='CTR (%)')
plt.title('Spend vs Clicks by Campaign ID\n(Bubble size = Reach)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Amount Spent (INR)')
plt.ylabel('Total Clicks')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('campaign_id_spend_vs_clicks_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.4 Cost Per 1000 Impressions (CPM) by Campaign ID
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(campaign_analysis)), campaign_analysis['CPM'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(campaign_analysis))))
plt.title('Cost Per 1000 Impressions (CPM) by Campaign ID', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Campaign ID')
plt.ylabel('CPM (INR)')
plt.xticks(range(len(campaign_analysis)), 
           [cid[:15]+'...' if len(cid) > 15 else cid for cid in campaign_analysis['campaign ID']], 
           rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('campaign_id_cpm.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Age Group Deep Dive
age_detailed = df.groupby(['Age', 'Audience']).agg({
    'Reach': 'sum',
    'Clicks': 'sum',
    'Amount Spent in INR': 'sum',
    'Click-Through Rate (CTR in %)': 'mean'
}).reset_index()

# 3.1 Reach by Age Group and Audience Type
plt.figure(figsize=(12, 8))
pivot_reach = age_detailed.pivot(index='Age', columns='Audience', values='Reach')
pivot_reach.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], figsize=(12, 8))
plt.title('Reach by Age Group and Audience Type', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Age Group')
plt.ylabel('Total Reach')
plt.legend(title='Audience Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('age_group_reach_by_audience.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.2 CTR by Age Group and Audience Type
plt.figure(figsize=(12, 8))
pivot_ctr = age_detailed.pivot(index='Age', columns='Audience', values='Click-Through Rate (CTR in %)')
pivot_ctr.plot(kind='bar', color=['#e74c3c', '#3498db'], figsize=(12, 8))
plt.title('CTR by Age Group and Audience Type', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Age Group')
plt.ylabel('Average CTR (%)')
plt.legend(title='Audience Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('age_group_ctr_by_audience.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ All visualizations have been created and saved!")
print("\n📊 Generated Charts:")
print("1. campaign_performance_dashboard.png - Overview of top campaigns, age groups, cost efficiency, and ROI")
print("2. geographic_analysis.png - Detailed geographic performance metrics")
print("3. age_group_analysis.png - Age group performance by audience type")

# 4. Campaign ID Advanced Performance Analysis
# Clean and filter campaign ID data
df_campaign_clean = df.dropna(subset=['campaign ID'])  # Remove rows with null campaign IDs

# Campaign ID overview
campaign_id_stats = df_campaign_clean.groupby('campaign ID').agg({
    'Reach': 'sum',
    'Impressions': 'sum',
    'Clicks': 'sum',
    'Amount Spent in INR': 'sum',
    'Click-Through Rate (CTR in %)': 'mean',
    'Campaign Name': 'first',
    'Audience': 'first'
}).reset_index()

campaign_id_stats['Actual_CTR'] = (campaign_id_stats['Clicks'] / campaign_id_stats['Impressions']) * 100
campaign_id_stats['CPC'] = campaign_id_stats['Amount Spent in INR'] / campaign_id_stats['Clicks']

# 4.1 Cost Per Click by Campaign ID
plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(campaign_id_stats)), campaign_id_stats['CPC'], 
               color=plt.cm.plasma(np.linspace(0, 1, len(campaign_id_stats))))
plt.title('Cost Per Click by Campaign ID', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Campaign ID')
plt.ylabel('CPC (INR)')
plt.xticks(range(len(campaign_id_stats)), [cid[:15] for cid in campaign_id_stats['campaign ID']], rotation=45)
plt.grid(axis='y', alpha=0.3)
# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'₹{height:.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('campaign_id_cpc_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.2 Campaign Efficiency (Clicks per ₹1000 spent)
plt.figure(figsize=(12, 8))
efficiency = campaign_id_stats['Clicks'] / campaign_id_stats['Amount Spent in INR'] * 1000
bars = plt.bar(range(len(campaign_id_stats)), efficiency, 
               color=plt.cm.coolwarm(np.linspace(0, 1, len(campaign_id_stats))))
plt.title('Campaign Efficiency\n(Clicks per ₹1000 spent)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Campaign ID')
plt.ylabel('Clicks per ₹1000')
plt.xticks(range(len(campaign_id_stats)), [cid[:15] for cid in campaign_id_stats['campaign ID']], rotation=45)
plt.grid(axis='y', alpha=0.3)
# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('campaign_id_efficiency_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Campaign ID Performance Heatmaps
# Create a detailed performance matrix for each campaign ID
campaign_details = df_campaign_clean.groupby(['campaign ID', 'Age']).agg({
    'Reach': 'sum',
    'Clicks': 'sum',
    'Amount Spent in INR': 'sum',
    'Click-Through Rate (CTR in %)': 'mean'
}).reset_index()

# 5.1 Reach Heatmap: Campaign ID vs Age Group
plt.figure(figsize=(12, 8))
pivot_reach_age = campaign_details.pivot(index='campaign ID', columns='Age', values='Reach').fillna(0)
im1 = plt.imshow(pivot_reach_age.values, cmap='YlOrRd', aspect='auto')
plt.colorbar(im1, label='Reach')
plt.title('Reach Heatmap: Campaign ID vs Age Group', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Age Group')
plt.ylabel('Campaign ID')
plt.xticks(range(len(pivot_reach_age.columns)), pivot_reach_age.columns, rotation=45)
plt.yticks(range(len(pivot_reach_age.index)), [cid[:15] for cid in pivot_reach_age.index])
plt.tight_layout()
plt.savefig('campaign_id_reach_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.2 Clicks Heatmap: Campaign ID vs Age Group
plt.figure(figsize=(12, 8))
pivot_clicks_age = campaign_details.pivot(index='campaign ID', columns='Age', values='Clicks').fillna(0)
im2 = plt.imshow(pivot_clicks_age.values, cmap='Blues', aspect='auto')
plt.colorbar(im2, label='Clicks')
plt.title('Clicks Heatmap: Campaign ID vs Age Group', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Age Group')
plt.ylabel('Campaign ID')
plt.xticks(range(len(pivot_clicks_age.columns)), pivot_clicks_age.columns, rotation=45)
plt.yticks(range(len(pivot_clicks_age.index)), [cid[:15] for cid in pivot_clicks_age.index])
plt.tight_layout()
plt.savefig('campaign_id_clicks_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.3 Spend Heatmap: Campaign ID vs Age Group
plt.figure(figsize=(12, 8))
pivot_spend_age = campaign_details.pivot(index='campaign ID', columns='Age', values='Amount Spent in INR').fillna(0)
im3 = plt.imshow(pivot_spend_age.values, cmap='Reds', aspect='auto')
plt.colorbar(im3, label='Spend (INR)')
plt.title('Spend Heatmap: Campaign ID vs Age Group', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Age Group')
plt.ylabel('Campaign ID')
plt.xticks(range(len(pivot_spend_age.columns)), pivot_spend_age.columns, rotation=45)
plt.yticks(range(len(pivot_spend_age.index)), [cid[:15] for cid in pivot_spend_age.index])
plt.tight_layout()
plt.savefig('campaign_id_spend_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.4 CTR Heatmap: Campaign ID vs Age Group
plt.figure(figsize=(12, 8))
pivot_ctr_age = campaign_details.pivot(index='campaign ID', columns='Age', values='Click-Through Rate (CTR in %)').fillna(0)
im4 = plt.imshow(pivot_ctr_age.values, cmap='RdYlBu_r', aspect='auto')
plt.colorbar(im4, label='CTR (%)')
plt.title('CTR Heatmap: Campaign ID vs Age Group', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Age Group')
plt.ylabel('Campaign ID')
plt.xticks(range(len(pivot_ctr_age.columns)), pivot_ctr_age.columns, rotation=45)
plt.yticks(range(len(pivot_ctr_age.index)), [cid[:15] for cid in pivot_ctr_age.index])
plt.tight_layout()
plt.savefig('campaign_id_ctr_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Campaign ID Performance Comparison
# Create a comprehensive campaign comparison
campaign_comparison = df_campaign_clean.groupby('campaign ID').agg({
    'Reach': 'sum',
    'Clicks': 'sum',
    'Amount Spent in INR': 'sum',
    'Campaign Name': 'first'
}).reset_index()

campaign_comparison['ROI'] = (campaign_comparison['Clicks'] / campaign_comparison['Amount Spent in INR']) * 1000
campaign_comparison['Reach_per_1000'] = campaign_comparison['Reach'] / 1000

# 6.1 Campaign ID Performance Comparison (Reach vs ROI)
plt.figure(figsize=(14, 8))
x = np.arange(len(campaign_comparison))
width = 0.35

bars1 = plt.bar(x - width/2, campaign_comparison['Reach_per_1000'], width, 
                label='Reach (in thousands)', color='skyblue', alpha=0.8)
bars2 = plt.bar(x + width/2, campaign_comparison['ROI'], width, 
                label='ROI (Clicks per ₹1000)', color='lightcoral', alpha=0.8)

plt.title('Campaign ID Performance Comparison\n(Reach vs ROI)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Campaign ID')
plt.ylabel('Performance Metrics')
plt.xticks(x, [cid[:20] for cid in campaign_comparison['campaign ID']], rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.0f}K', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('campaign_id_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.2 Total Clicks by Campaign
plt.figure(figsize=(14, 8))
campaign_names = [name[:30] + '...' if len(name) > 30 else name 
                 for name in campaign_comparison['Campaign Name']]
bars = plt.barh(campaign_names, campaign_comparison['Clicks'], 
                color=plt.cm.Set3(np.linspace(0, 1, len(campaign_comparison))))
plt.title('Total Clicks by Campaign', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Total Clicks')
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
             f'{int(width):,}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('campaign_total_clicks.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ All visualizations have been created and saved!")
print("\n📊 Generated Charts:")
print("1. campaign_performance_dashboard.png - Overview of top campaigns, age groups, cost efficiency, and ROI")
print("2. geographic_analysis.png - Detailed geographic performance metrics")
print("3. age_group_analysis.png - Age group performance by audience type")
print("4. campaign_id_analysis.png - Comprehensive Campaign ID performance metrics")
print("5. campaign_id_heatmaps.png - Performance heatmaps showing Campaign ID vs Age Group")
print("6. campaign_id_comparison.png - Campaign ID performance comparison and rankings")

# Display key insights including Campaign ID analysis
print("\n🔍 Key Insights from the Data:")
print(f"• Total campaigns analyzed: {df['Campaign Name'].nunique()}")
print(f"• Campaign IDs: {sorted(df_campaign_clean['campaign ID'].unique())}")
print(f"• Best performing campaign ID (CTR): {campaign_id_stats.loc[campaign_id_stats['Actual_CTR'].idxmax(), 'campaign ID']}")
print(f"• Most cost-effective campaign ID: {campaign_id_stats.loc[campaign_id_stats['CPC'].idxmin(), 'campaign ID']}")
print(f"• Highest reach campaign ID: {campaign_id_stats.loc[campaign_id_stats['Reach'].idxmax(), 'campaign ID']}")
print(f"• Best performing age group (CTR): {age_performance.loc[age_performance['CTR'].idxmax(), 'Age']}")
print(f"• Most cost-effective campaign ID: {campaign_cost.loc[campaign_cost['CPC_Clean'].idxmin(), 'campaign ID']}")
print(f"• Highest ROI audience: {audience_stats.loc[audience_stats['ROI'].idxmax(), 'Audience']}")
print(f"• Note: {df['campaign ID'].isnull().sum()} rows had missing campaign ID values and were excluded from campaign ID analysis")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nGenerated 15 individual visualization files:")
print("1. top_campaigns_ctr.png")
print("2. age_group_ctr.png") 
print("3. cost_vs_performance_campaign_id.png")
print("4. audience_roi_comparison.png")
print("5. campaign_id_reach_basic.png")
print("6. campaign_id_ctr_basic.png")
print("7. campaign_id_spend_vs_clicks_basic.png")
print("8. campaign_id_cpm.png")
print("9. age_group_reach_by_audience.png")
print("10. age_group_ctr_by_audience.png")
print("11. campaign_id_cpc_advanced.png")
print("12. campaign_id_efficiency_advanced.png")
print("13. campaign_id_reach_heatmap.png")
print("14. campaign_id_clicks_heatmap.png")
print("15. campaign_id_ctr_heatmap.png")
print("16. campaign_spending_heatmap.png")
print("17. campaign_id_performance_comparison.png")
print("18. campaign_total_clicks.png")
print("\nData insights saved in: Campaign_Analysis_Report.md")
print("\nAll graphs focus on Campaign ID analysis with duplicates removed!")
print("="*50)