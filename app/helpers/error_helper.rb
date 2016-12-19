module ErrorHelper
  def self.mail_error(msg, error)
    ErrorMailer.notify_error("#{msg} - #{error}")
  end
end