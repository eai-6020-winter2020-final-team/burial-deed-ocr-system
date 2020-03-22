let fields = [];

function displayImage(a){
	let imageDisplay = $('#image_display');
	imageDisplay.attr('src', '');
	let id = $(a).parent().parent().attr('id');
	imageDisplay.attr('src', '/getimage/' + id);
	imageDisplay.fadeIn(150);
	imageDisplay.click(() => {
		imageDisplay.fadeOut(150);
	});
}

function putData(d){
	let table = $('#query_table');
	let trHead = $('#table_headline');
	if(d.length === 0){
		$('p').text("This table is empty");
		return;
	}
	//draw head line, record fields
	for(let i of Object.keys(d[0])){
		switch(i){
			case 'id': break;
			case 'filename':
				trHead.append($('<th>').text('File Name'));
				break;
			default:
				fields.push(i);
				trHead.append($('<th>').text(i));
		}
	}
	//draw table body
	for(let record of d){
		let newTr = $('<tr>');
		for(let key of Object.keys(record)){
			switch(key){
				case 'id':
					newTr.attr('id', record[key]);
					break;
				case 'filename':
					let imgLink = $('<a>').text(record[key]).attr('onclick', 'displayImage(this)');
					newTr.append($('<td>').append(imgLink).addClass('filename_td'));
					break;
				default:
					newTr.append($('<td>').text(record[key]));
			}
		}
		//TODO: Edit and Delete
		let btnTd = $('<td>');
		btnTd.append($('<div>').text('Edit').addClass('table_btn edit_btn').click(clickEdit));
		btnTd.append($('<div>').text('Delete').addClass('table_btn delete_btn').click(clickDelete));
		btnTd.append($('<div>').text('Confirm').addClass('table_btn conf_btn').click(clickConf));
		btnTd.append($('<div>').text('Discard').addClass('table_btn disc_btn').click(clickDisc));
		newTr.append(btnTd);
		table.append(newTr);
	}
}

function clickEdit(){
	let btnTd = $(this).parent();
	btnTd.attr('current', 'edit');
	let editTds = btnTd.prevUntil('.filename_td');
	for(let editTd of editTds){
		let tdVal = $(editTd).text();
		let tmpInput = $('<input>').attr('type', 'text').attr('preVal', tdVal).val(tdVal).addClass('edit_input');
		$(editTd).html(tmpInput);
	}
	$(this).hide();
	$(this).siblings('.delete_btn').hide();
	$(this).siblings('.disc_btn').css('display', 'inline-block');
	$(this).siblings('.conf_btn').css('display', 'inline-block');
}

function clickDelete(){
	let btnTd = $(this).parent();
	btnTd.attr('current', 'delete');
	btnTd.parent().css('background-color', '#e64b37');
	$(this).hide();
	$(this).siblings('.edit_btn').hide();
	$(this).siblings('.disc_btn').css('display', 'inline-block');
	$(this).siblings('.conf_btn').css('display', 'inline-block');
}

function clickConf(){
	let btnClicked = $(this);
	let btnTd = $(this).parent();
	if(btnTd.attr('current') === 'edit'){
		let editTds = btnTd.prevUntil('.filename_td');
		let para = {};
		let paraLength = editTds.length;
		for(let i = 0; i < paraLength; i++){
			para[fields[paraLength - i - 1]] = $(editTds[i]).children('input').val();
		}
		para['id'] = btnTd.parent().attr('id');
		console.log(para);
		$.post(
			'/editrecord/',
			para
		).done(() => {
			for(let editTd of editTds){
				let tdVal = $(editTd).children('input').val();
				$(editTd).empty();
				$(editTd).text(tdVal);
			}
			btnTd.removeAttr('current');
			btnClicked.hide();
			btnClicked.siblings('.disc_btn').hide();
			btnClicked.siblings('.edit_btn').css('display', 'inline-block');
			btnClicked.siblings('.delete_btn').css('display', 'inline-block');
			btnTd.parent().removeClass('animated');
			btnTd.parent().addClass('animated');
		}).fail(() => {
			alert("Edit failed");
			btnClicked.siblings('.disc_btn').click();
		});
	} else {
		let para = {};
		para['id'] = btnTd.parent().attr('id');
		$.post(
			'/deleterecord/',
			para
		).done(() => {
			btnTd.parent().fadeOut();
		}).fail(() => {
			alert("Delete failed");
			btnClicked.siblings('.disc_btn').click();
		});
	}
}

function clickDisc(){
	let btnTd = $(this).parent();
	if(btnTd.attr('current') === 'edit'){
		let editTds = btnTd.prevUntil('.filename_td');
		for(let editTd of editTds){
			let tdVal = $(editTd).children('input').attr('preVal');
			$(editTd).empty();
			$(editTd).text(tdVal);
		}
	} else {
		btnTd.parent().css('background-color', '');
	}
	btnTd.removeAttr('current');
	$(this).hide();
	$(this).siblings('.conf_btn').hide();
	$(this).siblings('.edit_btn').css('display', 'inline-block');
	$(this).siblings('.delete_btn').css('display', 'inline-block');
}

$(document).ready(() => {
	$.getJSON('/getrecords/', {RecordType: recordType}, putData);
	$('.edit_btn').click(() => {console.log(this)});
});