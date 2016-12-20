module ErrorHelper
  def self.mail_error(msg, error_msg)
    ErrorMailer.notify_error("#{msg} - #{error_msg}").deliver
  end
end