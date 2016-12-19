class ErrorMailer < ApplicationMailer
  def notify_error(msg)
    @msg = msg
    mail(to: Rails.application.secrets.error_to_email, subject: 'Error in Fuel')
  end
end