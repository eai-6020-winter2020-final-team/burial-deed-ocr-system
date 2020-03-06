var trCounter = 0;

function fileSelected(file){
	var fileName = $(file).val();
	var selectButton = $("#select_button")[0];
	var previewDiv = $("#preview")[0];
	if (!fileName || !(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg"))){
		selectButton.innerHTML = '<span style="color:red">Only JPEG images are supported</span>';
		previewDiv.innerHTML = "";
		return false;
	}
	
	selectButton.innerHTML = file.files[0].name;
	if(file.files && file.files[0]){
		var reader = new FileReader();
		reader.onload = function(evt){
			previewDiv.innerHTML = '<img src="' + evt.target.result + '"/>';
		};
		reader.readAsDataURL(file.files[0]);
	}
}

$(document).ready(function(){
	$("#select_button").click(function(){
		$("input").click();
	});
	
	$("#submit_button").click(function(){
		var fileName = $("#file_uploader").val();
		var selectButton = $("#select_button")[0];
		if(!fileName){
			selectButton.innerHTML = '<span style="color:red">Please Select A Image</span>';
			return false;
		}
		if (!(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg"))){
			alert("Only JPEG images are supported");
			return false;
		}
		
		var formData = new FormData();
		var file = $("#file_uploader")[0].files[0];
		formData.append("image_file", file);
		$.ajax({
			type: "POST",
			url: "/upload/",
			data: formData,
			processData: false,
			contentType: false,
			success: function(successData){
				$("table").css("display", "inline-block");
				jsonData = JSON.parse(successData);
				var newTr = "";
				if(++trCounter % 2){
					newTr += "<tr>";
				}else{
					newTr += '<tr style="background-color: #EDF5D6;">';
				}
				newTr += "<td>";
				newTr += jsonData.fileName;
				newTr += "</td>";
				newTr += "<td>";
				newTr += jsonData.personName;
				newTr += "</td>";
				newTr += "<td>";
				newTr += jsonData.outputV1;
				newTr += "</td>";
				newTr += "<td>";
				newTr += jsonData.outputV2;
				newTr += "</td>";
				newTr += "<td>";
				newTr += jsonData.outputV3;
				newTr += "</td>";
				newTr += "<td>";
				newTr += jsonData.outputV4;
				newTr += "</td>";
				newTr += "</tr>"
				$("table").append($(newTr));
			}
		});
	});
});