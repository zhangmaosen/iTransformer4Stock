![alt text](image.png)


## fan
fans() {
        nvidia-settings --display :1.0 -a "[gpu:0]/GPUFanControlState=1" -a "[fan:0]/GPUTargetFanSpeed=$1"
        echo "Fan speeds set to $1 percent"
}