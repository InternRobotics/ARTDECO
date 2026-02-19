# PSNR
# Onthefly
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/acepro2/VID*/onthefly/ds_2-vanilla/metadata.json > results/psnrs/08_05_acepro2-onthefly.txt 
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/canon-r8/MVI*/onthefly/ds_2-vanilla/metadata.json > results/psnrs/08_05_canon-r8-onthefly.txt 
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/usbcam/*f/run*/onthefly/ds_2-vanilla/metadata.json > results/psnrs/08_05_usbcam-onthefly.txt 

echo "Onthefly PSNR"
echo "acepro2"; python scripts/avg.py results/psnrs/08_05_acepro2-onthefly.txt
echo "canon-r8"; python scripts/avg.py results/psnrs/08_05_canon-r8-onthefly.txt
echo "usbcam"; python scripts/avg.py results/psnrs/08_05_usbcam-onthefly.txt
echo;

# Artdeco
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/acepro2/VID*/artdeco/ds_2-lr_poses_0/metadata.json > results/psnrs/08_05_acepro2-artdeco.txt 
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/canon-r8/MVI*/artdeco/ds_2-lr_poses_0/metadata.json > results/psnrs/08_05_canon-r8-artdeco.txt 
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/usbcam/*f/run*/artdeco/ds_2-lr_poses_0/metadata.json > results/psnrs/08_05_usbcam-artdeco.txt 

echo "Artdeco PSNR"
echo "acepro2"; python scripts/avg.py results/psnrs/08_05_acepro2-artdeco.txt
echo "canon-r8"; python scripts/avg.py results/psnrs/08_05_canon-r8-artdeco.txt
echo "usbcam"; python scripts/avg.py results/psnrs/08_05_usbcam-artdeco.txt
echo;

# Gsplat
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/acepro2/VID*/gsplat/stats/val_step29999.json > results/psnrs/08_05_acepro2-gsplat.txt 
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/canon-r8/MVI*/gsplat/stats/val_step29999.json > results/psnrs/08_05_canon-r8-gsplat.txt 
grep -rH "PSNR" results/shared_results/zzw_results/l1/08_05/usbcam/*f/run*/gsplat/stats/val_step29999.json > results/psnrs/08_05_usbcam-gsplat.txt 


# time
# Onthefly
grep -rH "time" results/shared_results/zzw_results/l1/08_05/acepro2/VID*/onthefly/ds_2-vanilla/metadata.json > results/times/08_05_acepro2-onthefly.txt 
grep -rH "time" results/shared_results/zzw_results/l1/08_05/canon-r8/MVI*/onthefly/ds_2-vanilla/metadata.json > results/times/08_05_canon-r8-onthefly.txt 
# grep -rH "time" results/shared_results/zzw_results/l1/08_05/usbcam/*f/run*/onthefly/ds_2-vanilla/metadata.json > results/times/08_05_usbcam-onthefly.txt 
# Artdeco
grep -rH "time" results/shared_results/zzw_results/l1/08_05/acepro2/VID*/artdeco/ds_2-lr_poses_0/metadata.json > results/times/08_05_acepro2-artdeco.txt 
grep -rH "time" results/shared_results/zzw_results/l1/08_05/canon-r8/MVI*/artdeco/ds_2-lr_poses_0/metadata.json > results/times/08_05_canon-r8-artdeco.txt 
# grep -rH "time" results/shared_results/zzw_results/l1/08_05/usbcam/*f/run*/artdeco/ds_2-lr_poses_0/metadata.json > results/times/08_05_usbcam-artdeco.txt 
# Gsplat
grep -rH "time" results/shared_results/zzw_results/l1/08_05/acepro2/VID*/gsplat/stats/val_step29999.json > results/times/08_05_acepro2-gsplat.txt 
grep -rH "time" results/shared_results/zzw_results/l1/08_05/canon-r8/MVI*/gsplat/stats/val_step29999.json > results/times/08_05_canon-r8-gsplat.txt 
# grep -rH "time" results/shared_results/zzw_results/l1/08_05/usbcam/*f/run*/gsplat/stats/val_step29999.json > results/times/08_05_usbcam-gsplat.txt 
