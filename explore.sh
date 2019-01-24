# 1. Lower bound threshold (minimal interference)
echo Lower bound threshold \(minimal interference\)
python visual_pathway.py -amy_s 0.5 -vmpfc_s 0.5 -av 1.0 -va 1.0 -l 1.0 -n 1 -f log/1_lower.png > log/1_lower.log

# 2. Sensitive amygdala
echo Sensitive amygdala
python visual_pathway.py -amy_s 0.65 -vmpfc_s 0.5 -av 1.0 -va 1.0 -l 1.0 -n 1 -f log/2_sensitive.png > log/2_sensitive.log

# 3. Decreased top-down inhibition
echo Decreased top-down inhibition
python visual_pathway.py -amy_s 0.5 -vmpfc_s 0.5 -av 1.0 -va 0.85 -l 1.0 -n 1 -f log/3_topdown.png > log/3_topdown.log

# 4. Both
echo Both
python visual_pathway.py -amy_s 0.65 -vmpfc_s 0.5 -av 1.0 -va 0.85 -l 1.0 -n 1 -f log/4_both.png > log/4_both.log

# 5. Compensated Lower bound threshold (minimal interference)
echo Lower bound threshold \(minimal interference\)
python visual_pathway.py -amy_s 0.5 -vmpfc_s 0.5 -av 1.0 -va 1.0 -l 1.5 -n 1 -f log/5_compensated_lower.png > log/5_compensated_lower.log

# 6. Compensated Sensitive amygdala
echo Sensitive amygdala
python visual_pathway.py -amy_s 0.65 -vmpfc_s 0.5 -av 1.0 -va 1.0 -l 1.5 -n 1 -f log/6_compensated_sensitive.png > log/6_compensated_sensitive.log

# 7. Compensated Decreased top-down inhibition
echo Decreased top-down inhibition
python visual_pathway.py -amy_s 0.5 -vmpfc_s 0.5 -av 1.0 -va 0.85 -l 1.5 -n 1 -f log/7_compensated_topdown.png > log/7_compensated_topdown.log

# 8. Compensated Both
echo Compensation by lpfc
python visual_pathway.py -amy_s 0.65 -vmpfc_s 0.5 -av 1.0 -va 0.85 -l 1.5 -n 1 -f log/8_compensated_both.png > log/8_compensated_both.log
