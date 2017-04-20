sudo nvidia-smi --persistence-mode=1 -i 2
sudo nvidia-smi --applications-clocks-permission=0 -i 2
sudo nvidia-smi --query-supported-clocks="mem","gr" --format=csv -i 2
sudo nvidia-smi --applications-clocks=3003,1531 -i 2 #p4
sudo nvidia-smi --auto-boost-permission=0 -i 2
sudo nvidia-smi --auto-boost-default=0 -i 2
