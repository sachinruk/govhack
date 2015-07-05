

(function (_this, $) {


    _this.initialise = function () {
        $(function () {
            bindSignUpButton();
        });

        return _this;
    }

    var bindSignUpButton = function() {
    	$('#signUpSection .button').on('click', function() {
    		$('#signUpSection .signUp').slideUp();
    		$('#signUpSection .signedUp').slideDown();
    	});
    }
 
    // Initialise & assign to global scope
    window.Footer = _this.initialise();
})(window.Footer || {}, jQuery);