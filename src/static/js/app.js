
// Get the progress bar element
const progressBar = document.querySelector('.progress-bar');

// Fetch the progress percentage from the server
async function getProgress() {
  const response = await fetch('/progress');
  const data = await response.json();
  return data.progress;
}

// Update the progress bar width and color based on the progress percentage
async function updateProgressBar() {
  const progress = await getProgress();
  const progressWidth = `${progress}%`;
  const progressColor = getProgressColor(progress);
  progressBar.querySelector('.progress').style.width = progressWidth;
  progressBar.querySelector('.progress').style.backgroundColor = progressColor;
  progressBar.querySelector('.progress-text').textContent = progressWidth;
}

// Map progress percentage to color
function getProgressColor(progress) {
  if (progress >= 0 && progress <= 20) {
    return 'red';
  } else if (progress > 20 && progress <= 40) {
    return 'orange';
  } else if (progress > 40 && progress <= 60) {
    return 'yellow';
  } else if (progress > 60 && progress <= 80) {
    return 'limegreen';
  } else {
    return 'green';
  }
}

// Update the progress bar every 5 seconds
setInterval(updateProgressBar, 5000);



window.addEventListener('load', function() {
    const progress = document.querySelector('.progress');
    progress.style.animation = 'fill-progress 1s ease-in-out forwards';
  });
  
























const shoes = document.querySelectorAll('.shoe')




const selectImage = document.querySelector('.select-image')
const inputFile = document.querySelector('#file')
const imgArea = document.querySelector('.img-area')
const iconArea = document.querySelector('.icon')

const shoeBg = document.querySelector('.shoeBackground')


selectImage.addEventListener('click', function (){
    inputFile.click();  
   
})



inputFile.addEventListener('change', function () {

    const image = this.files[0]
    console.log(image);
    const reader = new FileReader();

    reader.onload = ()=> {

        const allImg = imgArea.querySelectorAll('img');
        allImg.forEach(item=> item.remove())
        const imgUrl = reader.result;
        const img = document.createElement('img');
        img.src = imgUrl;
        setTimeout(() => {

            imgArea.appendChild(img);
            imgArea.dataset.img = image.name;

            selectImage.classList.remove('loading');


        }, 2000)

    }
    reader.readAsDataURL(image)
})





let x = window.matchMedia("(max-width: 1000px)");

function changeHeight(){
    if(x.matches){
        let shoeHeight = shoes[0].offsetHeight;
        shoeBg.getElementsByClassName.height = `${shoeHeight * 0.9}px`
    }
    else{
        shoeBg.style.height = "475px"
    }
}

changeHeight()
window.addEventListener('resize', changeHeight)