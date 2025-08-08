# AQI Pipeline - Simple Configuration

## ✅ **What You Have Now**

### **Simple, Clean Setup**
- **No complex strategy options** - just one optimized approach
- **1-day retraining threshold** - triggers after ~23 new records (≈24 hours)
- **Full data retraining** - always uses complete dataset for best accuracy
- **Daily automation** - runs every day at 2:00 AM UTC

### **What Happens After Push**

1. **First Run**: Full training on all historical data (no existing models)
2. **Daily Runs**: Only retrain when ≥23 new records are available
3. **Training**: Always uses ALL data for maximum accuracy and context preservation

### **Manual Testing**
```bash
# Test specific model
python ci_cd/train_model_pipeline.py --horizon 24h

# Force retrain (bypass data freshness check)
FORCE_RETRAIN=true python ci_cd/train_model_pipeline.py --horizon 24h
```

### **GitHub Actions**
- **Automatic**: Daily at 2:00 AM UTC
- **Manual**: Go to Actions → "Smart AQI Model Training" → Run workflow
- **Options**: Select horizon (24h/48h/72h/all) and force retrain (yes/no)

## 🎯 **Perfect for Your Needs**
- ✅ **Simple**: No complex configurations
- ✅ **Efficient**: Only trains when needed (daily data threshold)  
- ✅ **Accurate**: Always uses full dataset context
- ✅ **Automated**: Runs daily without intervention
- ✅ **Flexible**: Manual trigger available for testing

**You're ready to push and deploy!** 🚀
