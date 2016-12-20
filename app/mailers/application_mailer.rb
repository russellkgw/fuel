class ApplicationMailer < ActionMailer::Base
  default from: Rails.application.secrets.error_from_email
end