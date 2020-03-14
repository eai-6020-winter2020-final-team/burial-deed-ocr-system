function checkFileName(fileName){
	var supportedExtension = [".jpg", ".jpeg", ".png", ".tif", ".tiff"];
	for(var ext of supportedExtension){
		if(fileName.toLowerCase().endsWith(ext)) return true;
	}
	return false;
}

function fileSelected(file){
	var fileName = $(file).val();
	var selectButton = $("#select_button");
	var previewDiv = $("#preview");
	if (!fileName || !checkFileName(fileName)){
		selectButton.html($("<span>").text("Only JPEG/PNG/TIFF images are supported").css("color", "red"));
		previewDiv.empty();
		return false;
	}
	selectButton.html(file.files[0].name);
	if(file.files && file.files[0]){
		var reader = new FileReader();
		reader.onload = (evt) => {
			previewDiv.html($("<img>").attr("alt", "Selected Image").attr("src", evt.target.result));
		};
		reader.readAsDataURL(file.files[0]);
	}
}

$(document).ready(() => {
	$("#select_button").click(function(){
		$("input").click();
	});
	
	$("#submit").click(() => {
		var fileName = $("#file_uploader").val();
		var selectButton = $("#select_button");
		if(!fileName || !checkFileName(fileName)){
			selectButton.html($("<span>").text("Please Select an Image").css("color", "red"));
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
			success: (successData) => {
				var jsonData = JSON.parse(successData);
				$("table").css("display", "inline-block");
				var trHead = $("#table_headline");
				var tr1st = $("#table_1stline");
				trHead.empty();
				tr1st.empty();
				for(var i of Object.keys(jsonData)){
					trHead.append($("<th>").text(i));
					tr1st.append($("<td>").text(jsonData[i]));
				}
			},
			error: (jqXHR) => {
				if(jqXHR.status === 500){
					alert("ErrorCode 500: An error occurred when processing this image");
				}
			}
		});
	});
});