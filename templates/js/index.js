//declaring html elements.

const imgDiv = document.querySelector('.profile-pic-div');
const img = document.querySelector('#photo');
const file = document.querySelector('#file');
const uploadBtn = document.querySelector('#uploadBtn');
const uploadimgBtn = document.querySelector('#uploadimgbtn');

//if user hover on profile div.

imgDiv.addEventListener('mouseenter', function(){
    //uploadBtn.style.display = "block";
});

//if we hover out of profile div.

imgDiv.addEventListener('mouseleave', function(){
    //uploadBtn.style.display = "none";
});

//when we choose a photo to upload.

file.addEventListener('change', function(){
    const choosedFile = this.files[0];

    if(choosedFile){
        const reader = new FileReader();

        reader.addEventListener('load', function(){
            img.setAttribute('src', reader.result);
        });

        reader.readAsDataURL(choosedFile);
    }
});